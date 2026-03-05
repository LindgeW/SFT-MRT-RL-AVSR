import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import editdistance
import copy
import random
import numpy as np
import sys
import os
import jiwer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from avdataset import GRIDDataset, CMLRDataset, BucketBatchSampler
from avmodel import AVSRModel
from constants import *


class JointLoss(nn.Module):
    def __init__(self, mrt_alpha=0.05, dpo_beta=0.1):
        super().__init__()
        self.mrt_alpha = mrt_alpha
        self.dpo_beta = dpo_beta

    def compute_mrt_loss(self, log_probs, risks):
        """
        log_probs: [Batch, N_samples] (包含 N-best 和 GT)
        risks: [Batch, N_samples] (WER, GT 的 risk 为 0)
        """
        # 1. 归一化概率分布 (Risk Minimization 需要相对概率)
        # alpha 用于平滑分布
        probs = F.softmax(log_probs * self.mrt_alpha, dim=-1)
        # 2. 计算预期风险: sum(P(y) * Risk(y))
        loss = torch.sum(probs * risks, dim=-1)
        return loss.mean()

    def compute_dpo_loss(self, policy_logps, ref_logps):
        """
        policy_logps: tuple(chosen_logp, rejected_logp)
        ref_logps: tuple(ref_chosen_logp, ref_rejected_logp)
        """
        pi_chosen, pi_rejected = policy_logps
        ref_chosen, ref_rejected = ref_logps
        # DPO 核心公式
        pi_ratio = pi_chosen - pi_rejected
        ref_ratio = ref_chosen - ref_rejected
        logits = self.dpo_beta * (pi_ratio - ref_ratio)
        loss = -F.logsigmoid(logits).mean()
        return loss



class BaseTrainer:
    def __init__(self, model, optimizer, lr_scheduler=None, accumulate_step=1, device='cuda:0'):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = 0
        self.accumulate_step = accumulate_step
    
    def train_step(self, vid_inp, aud_inp, targets, vid_lens, aud_lens, tgt_lens):
        ''' 
        optimizer.zero_grad()
        #losses = model(vid_inp, aud_inp, targets, vid_lens, aud_lens, tgt_lens)
        losses = model(clean_vid_inp, noisy_vid_inp, clean_aud_inp, noisy_aud_inp, targets, vid_lens, aud_lens, tgt_lens)
        loss = losses['avsr'] + losses['drl']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        '''
        loss = self.model(vid_inp, aud_inp, targets, vid_lens, aud_lens, tgt_lens)[0]
        #loss = self.model(clean_vid_inp, noisy_vid_inp, clean_aud_inp, noisy_aud_inp, targets, vid_lens, aud_lens, tgt_lens)
        loss = loss / self.accumulate_step
        loss.backward()
        self.step += 1
        if self.step % self.accumulate_step == 0:
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.step = 0
        return {"avsr_loss": loss.data.item()}



class MRT_DPO_Trainer:
    def __init__(self, model, optimizer, tokenizer, lambda_ce=0.2, lambda_mrt=1.0, lambda_dpo=0.1, device='cuda:0'):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.tokenizer = tokenizer   # 用于计算 WER 时的解码
        self.pad_id, self.bos_id, self.eos_id = (tokenizer.index(x) for x in [PAD, BOS, EOS])
        print(f'PAD ID: {self.pad_id}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id}')
        self.device = device
        
        # 初始化参考模型 (Reference Model)
        # 深拷贝当前 SFT 模型并冻结
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        self.criterion = JointLoss(mrt_alpha=1.)
        self.weights = {'ce': lambda_ce, 'mrt': lambda_mrt, 'dpo': lambda_dpo}

    def get_batch_log_probs(self, model, tgt_tokens, enc_memory, src_lens, tgt_lens):
        """
            通用辅助函数：计算给定 Token 序列在模型下的 Log-Prob (Sum over sequence)
        """
        # Transformer Decoder Forward
        # tgt_tokens: [Batch, Len] (包含 <sos>, <eos>)
        # enc_memory: Encoder 输出 [Batch, T, Dim]
        
        # 假设 model.decoder_forward 返回 logits [Batch, Len, Vocab]
        # 注意：这里调用的是 Teacher Forcing 模式
        all_log_probs = model.decoder_forward(tgt_tokens, enc_memory, src_lens, tgt_lens)
        
        # Gather 对应 Token 的概率
        # input: <sos> A B C ...
        # target: A B C <eos> ...
        # 通常 Transformer Decoder 输出对齐是 shift right 的
        # 这里假设 logits 已经对应了我们要预测的位置
        
        # 简单处理：取 tokens[:, 1:] 作为目标，logits[:, :-1, :] 作为预测
        targets = tgt_tokens[:, 1:]
        preds_log_probs = all_log_probs[:, :-1, :]
        
        target_log_probs = torch.gather(preds_log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding (假设 pad_id = 0)
        pad_mask = (targets != 0).float()
        
        # Sum log probs over sequence length
        seq_log_probs = (target_log_probs * pad_mask).sum(dim=-1)
        
        return seq_log_probs

    def train_step(self, vid_inp, aud_inp, tgt_tokens, vid_lens, aud_lens, tgt_lens, n_best=5):
        """
        vid_inp: [B, C, T, H, W]
        aud_inp: [B, T_a, D]
        tgt_tokens: [B, L]
        """
        batch_size = vid_inp.size(0)
        
        # -------------------------------------------
        # Step 1: Encoder 前向 (只做一次)
        # -------------------------------------------
        # 联合编码 AV 特征
        enc_memory, src_lens = self.model.encode_av(vid_inp, aud_inp, vid_lens, aud_lens)

        # -------------------------------------------
        # Step 2: 采样与构建数据 (No Grad)
        # -------------------------------------------
        with torch.no_grad():
            # Beam Search 产生 N-best
            # 返回: List[List[Dict{'tokens': Tensor, 'text': str}]]
            #nbest_batch = self.model.generate(enc_memory, mem_mask, beam_size=n_best)
            nbest_batch, tgt_text = [], []
            nbest_output = self.model.generate(enc_memory, src_lens, self.bos_id, self.eos_id, max_dec_len=50, beam_size=n_best)
            for preds, tgts in zip(nbest_output, tgt_tokens):
                batch_out = []
                for pred in preds:
                    txt = ''.join([self.tokenizer[i] for i in pred.tolist() if i not in [self.pad_id, self.bos_id, self.eos_id]])
                    batch_out.append({'text': txt, 'tokens': torch.cat([torch.tensor([self.bos_id], device=pred.device), pred])})
                nbest_batch.append(batch_out)
                
                tgt_text.append(''.join([self.tokenizer[i] for i in tgts.tolist() if i not in [self.pad_id, self.bos_id, self.eos_id]]))
            
            mrt_tokens_list = [] # 用于 MRT
            mrt_risks_list = []  # 用于 MRT
            dpo_rejected_tokens = [] # 用于 DPO
            for b in range(batch_size):
                hyps = nbest_batch[b] # N 个候选
                ref_text = tgt_text[b]
                # 计算 WER 并找 Hard Negative
                wers = []
                current_batch_tokens = []
                for h in hyps:
                    # 计算 WER (Risk)
                    #wer = editdistance.eval(h['text'].split(), ref_text.split()) / (len(ref_text.split()) + 1e-8)
                    wers.append(jiwer.wer(ref_text, h['text']))
                    current_batch_tokens.append(h['tokens'])
                
                # --- 构建 DPO 数据 ---
                # 策略：选择 WER 最高的作为 Rejected (Hard Negative)
                # 也可以加逻辑：如果最高 WER 也是 0，则忽略此样本
                #worst_idx = torch.argmax(torch.tensor(wers)).item()
                #worst_idx = wers.index(max(wers))  # WER最高
                if len(set(wers)) <= 2:
                    worst_idx = wers.index(max(wers))  # WER最高
                else:
                    #worst_idx = wers.index(sorted(set(wers))[1])  # WER次低
                    #worst_idx = wers.index(random.choice(sorted(set(wers))[1:-1]))  # WER次低
                    worst_idx = wers.index(random.choice(sorted(set(x for x in wers if x != 0))[:3]))
                dpo_rejected_tokens.append(hyps[worst_idx]['tokens'])
                
                # --- 构建 MRT 数据 ---
                # 技巧：将 GT (Risk=0) 注入到 N-best 列表末尾，这样 MRT 会包含 N+1 个样本
                current_batch_tokens.append(tgt_tokens[b])   # 加入 GT
                wers.append(0.0)   # GT 的 Risk 为 0
                
                mrt_tokens_list.extend(current_batch_tokens)
                mrt_risks_list.extend(wers)

            # Pad 序列以进行批处理
            # mrt_inputs: [Batch * (N+1), Max_Len]
            mrt_inputs = pad_sequence(mrt_tokens_list, batch_first=True, padding_value=0).to(self.device)
            mrt_risks = torch.tensor(mrt_risks_list).view(batch_size, n_best + 1).to(self.device)
            # dpo_rejected: [Batch, Max_Len]
            dpo_rejected = pad_sequence(dpo_rejected_tokens, batch_first=True, padding_value=0).to(self.device)

        # -------------------------------------------
        # Step 3: Policy Model 计算 Log-Probs (With Grad)
        # -------------------------------------------
        # A. 计算 MRT 需要的所有序列概率 (N-best + GT)
        # 扩展 enc_memory: [B, T, D] -> [B * (N+1), T, D]
        mrt_mem_expanded = enc_memory.repeat_interleave(n_best + 1, dim=0)
        src_lens_expanded = src_lens.repeat_interleave(n_best + 1, dim=0)
        tgt_lens_expanded = tgt_lens.repeat_interleave(n_best + 1, dim=0)
        
        # 得到 [Batch * (N+1)]
        all_mrt_logps = self.get_batch_log_probs(self.model, mrt_inputs, mrt_mem_expanded, src_lens_expanded, tgt_lens_expanded)
        # Reshape -> [Batch, N+1]
        all_mrt_logps_view = all_mrt_logps.view(batch_size, n_best + 1)
        
        # B. 提取 DPO 需要的 Chosen 和 Rejected 概率
        # Chosen 就是 GT，已经算过了，在 MRT列表的最后一个
        policy_chosen_logps = all_mrt_logps_view[:, -1] 
        
        # Rejected 比较麻烦，它在 MRT 列表里的位置不固定
        # 为了代码简单，我们单独算一次 Rejected 的概率 (稍微多花点计算量，但逻辑清晰) 或者从 all_mrt_logps 中根据 worst_idx 提取
        policy_rejected_logps = self.get_batch_log_probs(self.model, dpo_rejected, enc_memory, src_lens, tgt_lens)

        # -------------------------------------------
        # Step 4: Reference Model 计算 Log-Probs (No Grad)
        # -------------------------------------------
        with torch.no_grad():
            ref_mem = self.ref_model.encode_av(vid_inp, aud_inp, vid_lens, aud_lens)[0]
            ref_chosen_logps = self.get_batch_log_probs(self.ref_model, tgt_tokens, ref_mem, src_lens, tgt_lens)
            ref_rejected_logps = self.get_batch_log_probs(self.ref_model, dpo_rejected, ref_mem, src_lens, tgt_lens)

        # -------------------------------------------
        # Step 5: 计算联合损失
        # -------------------------------------------
        # 1. MRT Loss
        loss_mrt = self.criterion.compute_mrt_loss(all_mrt_logps_view, mrt_risks)
        
        # 2. DPO Loss
        loss_dpo = self.criterion.compute_dpo_loss(
            (policy_chosen_logps, policy_rejected_logps),
            (ref_chosen_logps, ref_rejected_logps)
        )
        
        # 3. CE Loss (Regularization)
        # 我们可以直接利用 policy_chosen_logps (它是 GT 的序列 log_prob)
        # CE Loss = -mean(log_prob)
        # 注意要除以 Token 数量进行归一化，这里简化为 Seq 级别的 mean
        loss_ce = -policy_chosen_logps.mean()

        total_loss = self.weights['mrt'] * loss_mrt + self.weights['dpo'] * loss_dpo + self.weights['ce'] * loss_ce

        self.optimizer.zero_grad()
        
        total_loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "mrt": loss_mrt.item(),
            "dpo": loss_dpo.item(),
            "ce": loss_ce.item()
        }
        


def main():
    device = torch.device('cuda:' + str(sys.argv[1]))
    print('running device:', torch.cuda.get_device_name(), device)
    data_type = str(sys.argv[2]).strip().lower()  # grid or cmlr or lrs2/3
    if data_type == 'grid':
        data_root = r'../LipData/GRID/LIP_160_80/lip'
        train_set = GRIDDataset(data_root, r'data/unseen_train.json', phase='train', setting='unseen')
        #val_set = GRIDDataset(data_root, r'data/unseen_val.json', phase='test', setting='unseen')
    elif data_type == 'cmlr':
        data_root = r'../LipData/CMLR'
        train_set = CMLRDataset(data_root, r'data/unseen_train.csv', phase='train', setting='unseen')
        #val_set = CMLRDataset(data_root, r'data/unseen_test.csv', phase='test', setting='unseen')
    else:
        raise NotImplementedError('Unknown Dataset!!')
    
    #model_path = None
    model_path = 'grid_avg_10.pt'
    #model_path = 'checkpoints/grid/stage1_iter_49.pt'
    
    model = AVSRModel(len(train_set.vocab)).to(device)
    print('参数量(M)：', sum(param.numel() for param in model.parameters())/1e6)
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print(f'loading weights from {model_path} ...')
    print(model)

    batch_size = 32
    accumulate_step = 1
    epochs = {'I': 50, 'II': 5}   
    lrs = {'I': 3e-4, 'II': 1e-5}  # 1.61
    savedir = os.path.join('checkpoints', 'grid')

    train_set.noise_ratio = 0.25
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_set.collate_pad)  
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.AdamW([*model.avsr.parameters(), *model.spk.parameters()], lr=3 * lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lrs['I'], betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    num_iters = len(data_loader) * epochs['I'] // accumulate_step
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_iters // 10, num_training_steps=num_iters)
    ## I AVSR预训练
    base_trainer = BaseTrainer(model, optimizer, lr_scheduler, accumulate_step=accumulate_step, device=device)
    for ep in range(epochs['I']):
        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            #clean_vid_inp = batch_data['clean_vid'].to(device)
            #clean_aud_inp = batch_data['clean_aud'].to(device)
            noisy_vid_inp = batch_data['noisy_vid'].to(device)
            noisy_aud_inp = batch_data['noisy_aud'].to(device)
            targets = batch_data['txt'].to(device)
            vid_lens = batch_data['vid_lens'].to(device)
            aud_lens = batch_data['aud_lens'].to(device)
            tgt_lens = batch_data['txt_lens'].to(device)
            #print(batch_data['clean_vid_lens'], batch_data['noisy_vid_lens'], batch_data['clean_aud_lens'], batch_data['noisy_aud_lens'])
            loss_val = base_trainer.train_step(noisy_vid_inp, noisy_aud_inp, targets, vid_lens, aud_lens, tgt_lens)
            print(f'Epoch {ep}, Iter {i}, lr: {optimizer.param_groups[0]["lr"]}, loss: {loss_val["avsr_loss"]}', flush=True)
        if ep >= epochs['I'] - 10:
            savename = 'stage1_iter_{}.pt'.format(ep)
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)


    train_set.noise_ratio = 0.
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_set.collate_pad)  
    optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lrs['II'], betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    ## II MRT-DPO
    mrt_dpo_trainer = MRT_DPO_Trainer(model, optimizer2, train_set.vocab, lambda_mrt=0.8, lambda_dpo=0.2, device=device)
    for ep in range(epochs['II']):
        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            #clean_vid_inp = batch_data['clean_vid'].to(device)
            #clean_aud_inp = batch_data['clean_aud'].to(device)
            noisy_vid_inp = batch_data['noisy_vid'].to(device)
            noisy_aud_inp = batch_data['noisy_aud'].to(device)
            targets = batch_data['txt'].to(device)
            vid_lens = batch_data['vid_lens'].to(device)
            aud_lens = batch_data['aud_lens'].to(device)
            tgt_lens = batch_data['txt_lens'].to(device)
            #print(batch_data['clean_vid_lens'], batch_data['noisy_vid_lens'], batch_data['clean_aud_lens'], batch_data['noisy_aud_lens'])
            loss_val = mrt_dpo_trainer.train_step(noisy_vid_inp, noisy_aud_inp, targets, vid_lens, aud_lens, tgt_lens)
            print(f'Epoch {ep}, Iter {i}, lr: {optimizer2.param_groups[0]["lr"]}, loss: {loss_val["total"]}', flush=True)
        if ep > 1:
            savename = 'stage2_iter_{}.pt'.format(ep)
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)




@torch.no_grad()
def evaluate():
    device = torch.device('cuda:' + str(sys.argv[1]))
    print('running device:', torch.cuda.get_device_name(), device)
    data_type = str(sys.argv[2]).strip().lower()  # grid or cmlr or lrs2/3
    if data_type == 'grid':
        data_root = r'../LipData/GRID/LIP_160_80/lip'
        test_set = GRIDDataset(data_root, r'data/unseen_val.json', phase='test', setting='unseen')
    elif data_type == 'cmlr':
        data_root = r'../LipData/CMLR'
        test_set = CMLRDataset(data_root, r'data/unseen_test.csv', phase='test', setting='unseen')
    else:
        raise NotImplementedError('Unknown Dataset!!')
    
    batch_size = 32
    #model_path = 'grid_avg_10.pt'
    #model_path = 'checkpoints/grid/stage1_iter_49.pt'
    model_path = 'checkpoints/grid/stage2_iter_4.pt'
    
    model = AVSRModel(len(test_set.vocab)).to(device)
    print('参数量(M)：', sum(param.numel() for param in model.parameters())/1e6)
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print(f'loading weights from {model_path} ...')
    model.eval()
    print(len(test_set))

    data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=test_set.collate_pad)  
    preds, refs = [], []
    PAD_ID, BOS_ID, EOS_ID = (test_set.vocab.index(x) for x in [PAD, BOS, EOS])
    for batch_data in data_loader:
        #vid_inp = batch_data['vid'].to(device)
        #aud_inp = batch_data['aud'].to(device)
        vid_inp = batch_data['clean_vid'].to(device)
        #aud_inp = batch_data['clean_aud'].to(device)
        aud_inp = batch_data['noisy_aud'].to(device)
        tgt_txt = batch_data['txt'].to(device)
        #vid_lens = batch_data['clean_vid_lens'].to(device)
        #aud_lens = batch_data['clean_aud_lens'].to(device)
        vid_lens = batch_data['vid_lens'].to(device)
        aud_lens = batch_data['aud_lens'].to(device)
        #output = model.greedy_decode(vid_inp, input_lens)
        #vid_inp, aud_inp = model.avsr.recon_input(vid_inp, aud_inp, vid_lens, aud_lens)
        output = model.beam_search_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=50)
        for out, tgt in zip(output, tgt_txt):
            ## CER
            #preds.append(''.join([test_set.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            preds.append(''.join([test_set.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            refs.append(''.join([test_set.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            ## WER
            #preds.append(' '.join([test_set.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #preds.append(' '.join([test_set.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #refs.append(' '.join([test_set.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #print(preds[-1], '|||', refs[-1], preds[-1] == refs[-1])
            # write_to('pred-cmlr.txt', ref[-1]+'\t'+pred[-1]+'\t'+str(ref[-1] == pred[-1]))
    test_wer, test_cer = jiwer.wer(refs, preds), jiwer.cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}'.format(test_wer, test_cer))
    return test_wer, test_cer



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    
if __name__ == '__main__':
    set_seed(1337)
    main()
    #evaluate()


