# SFT-MRT-DPO learning for robust AVSR

The core idea is to combine MRT and DPO to train a robust AVSR model. The N-best candidate results decoded by beam search are used for MRT; the one with the highest or second-highest WER among the N results is taken as the rejected example, and the ground-truth is taken as the chosen example for DPO. But, before the MRT-DPO training, our AVSR model needs to undergo standard supervised training (i.e., SFT) to obtain the base/reference model.

- [x] Supervised Fine-Tuning (SFT)
- [x] Minimum Risk Training (MRT)
- [x] Direct Preference Optimization (DPO)

