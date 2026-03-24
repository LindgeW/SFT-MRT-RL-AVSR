# SFT-MRT-DPO/GRPO learning for robust AVSR

- The core idea is to combine MRT and DPO to train a robust AVSR model. The N-best candidate results decoded by beam search are used for MRT; the one with the highest or second-highest WER among the N results is taken as the rejected example, and the ground-truth is taken as the chosen example for DPO. Notably, before the MRT-DPO training, our AVSR model needs to undergo standard supervised training (i.e., SFT) to obtain the base/reference model.
- Improvement: replace DPO with GRPO; negative WER acts as reward; N-best results produce the group reward scores.
- WER = $\frac{(S + I + D)}{N}$ -> r = -WER / 1-WER / exp(-WER)
- type-aware WER = $\frac{(w_s\*S + w_i\*I + w_d\*D)}{N}$, $w_s/w_i/w_d$ control the importance of recognition error type.
  
- [x] Supervised Fine-Tuning (SFT)
- [x] Minimum Risk Training (MRT)
- [x] Direct Preference Optimization (DPO)
- [x] Group Relative Policy Optimization (GRPO)

