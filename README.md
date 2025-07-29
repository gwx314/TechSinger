# TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching

#### Wenxiang Guo, Yu Zhang, Changhao Pan, Rongjie Huang, Li Tang, Ruiqi Li, Zhiqing Hong, Yongqi Wang, Zhou Zhao | Zhejiang University

PyTorch Implementation of TechSinger (AAAI 2025): Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2502.12572)

We provide our implementation in this repository.

Visit our [demo page](https://gwx314.github.io/tech-singer/) for audio samples.

## News
- 2025.2: We released the checkpoints of TechSinger!
- 2025.2: We released the code of TechSinger!
- 2024.12: TechSinger is accepted by AAAI 2025!

## Quick Start
We provide an example of how you can generate high-fidelity samples using TechSinger.

To try on your own dataset or GTSinger, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.

### Pre-trained Models
Simply download the models from [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/verstar/TechSinger/tree/main).
Details of each folder are as follows:

| Model       |  Description                                                              | 
|-------------|--------------------------------------------------------------------------|
| stage1 |  stage1 model [(config)](./egs/stage1.yaml) |
| stage2 |  stage2 model [(config)](./egs/stage2.yaml) |
| HIFI-GAN    |  Neural Vocoder                       |

**Notably, this TechSinger checkpoint only supports Chinese and English! You should train your own model based on GTSinger for multilingual style transfer and control!**

### Dependencies

A suitable [conda](https://conda.io/) environment named `techsinger` can be created
and activated with:

```
conda create -n techsinger python=3.10
conda activate techsinger
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Multi-GPU

By default, this implementation uses as many GPUs in parallel as returned by `torch.cuda.device_count()`. 
You can specify which GPUs to use by setting the `CUDA_DEVICES_AVAILABLE` environment variable before running the training module.

## Inference for singing voices

Here we provide a singing synthesis pipeline using TechSinger.

1. Prepare **stage1, stage2**: Download and put checkpoint at `checkpoints/stage1`, `checkpoints/stage2`.
2. Prepare **HIFI-GAN**: Download and put checkpoint at `checkpoints/hifigan`.
3. Prepare **Information**: Provide singer id and input target ph, target note for each ph, target note_dur for each ph, target note_type for each ph (rest: 1, lyric: 2, slur: 3), and target technique for each ph (control: 0, technique: 1, random: 2). Input these information in `Inference/techsinger.py`. **Notably, if you want to use data from GTSinger to infer this checkpoint, you need to modify the phonemes in metadata.json of GTSinger (delete "_zh" or "_en" from each phoneme) to ensure that all phonemes are included in [phone_set](./ZHEN_checkpoint_phone_set.json)!**
4. Infer with techsinger:

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=$GPU python inference/techsinger.py --config egs/stage2.yaml  --exp_name stage2 --reset
```

5. You can also use cfg_scale to adjust the degree of the technique. The larger the scale value is, the stronger the degree of the technique will be. The default value is 1.0:

```bash
CUDA_VISIBLE_DEVICES=$GPU python inference/techsinger.py --config egs/stage2.yaml  --exp_name stage2 --hparams="cfg_scale=2.0"  --reset
```

Generated wav files are saved in `infer_out` by default.<br>

## Train your own model based on GTSinger

### Data Preparation 

1. Prepare your own singing dataset or download [GTSinger](https://github.com/GTSinger/GTSinger).
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types, and techniques for each singing voice) and `phone_set.json` (all phonemes of your dictionary) in `data/processed/tech` **(Note: GTSinger provides `metadata.json` and `phone_set.json`, but you need to change the wav_fn of each wav in `metadata.json` to your own absolute path)**.
3. Set `processed_data_dir` (`data/processed/tech`), `binary_data_dir`,`valid_prefixes` (list of parts of item names, like `["Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好"]`), `test_prefixes` in the [config](./egs/stage1.yaml).
4. Preprocess Dataset: 

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/bin/binarize.py --config egs/stage1.yaml
```

### Training TechSinger

1. Train Stage1 Model:
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stage1.yaml  --exp_name Stage1 --reset
```
2. Train Stage2 Model:
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stage2.yaml  --exp_name Stage2 --reset
```

### Inference with TechSinger

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stage2.yaml  --exp_name Stage2 --infer
```

## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[TCSinger](https://github.com/AaronZ345/TCSinger),
[GTSinger](https://github.com/AaronZ345/GTSinger)
as described in our code.

## Citations ##

If you find this code useful in your research, please cite our work:
```bib
@article{guo2025techsinger,
  title={TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching},
  author={Guo, Wenxiang and Zhang, Yu and Pan, Changhao and Huang, Rongjie and Tang, Li and Li, Ruiqi and Hong, Zhiqing and Wang, Yongqi and Zhao, Zhou},
  journal={arXiv preprint arXiv:2502.12572},
  year={2025}
}
```

## Disclaimer ##

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
