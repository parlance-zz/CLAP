# CLAP
<p align="center">
  <img src="https://raw.githubusercontent.com/LAION-AI/CLAP/main/assets/logo.PNG" alt="The Contrastive Language-Audio Pretraining Model Architecture" width="60%"/>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2211.06687"><img src="https://img.shields.io/badge/arXiv-2211.06687-brightgreen.svg?style=flat-square"/></a>
  <a href="https://pypi.org/project/laion-clap"><img src="https://badge.fury.io/py/laion-clap.svg"/></a>
  <a href="https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/clap"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue"/></a>
</p>
 
### This repository provides representations of audios and texts via Contrastive Language-Audio Pretraining (CLAP)

With CLAP, you can extract a latent representation of any given audio and text for your own model, or for different downstream tasks.

All codes are comming officially with the following paper, accepted by IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2023:
 - [Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687)

**New Updates:** 

<b>1. We release new CLAP pretrained checkpoints pretrained on music and speech data collecstions from [our dataset collection repo](https://github.com/LAION-AI/audio-dataset).</b>

<b>2. CLAP model is incorporated and supported by [HuggingFace Transformers](https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/clap). Many thanks to [Younes Belkada](https://huggingface.co/ybelkada) and [Arthur Zucker](https://fr.linkedin.com/in/arthur-zucker-8a0445144) for contributing to the HuggingFace support. </b>

## About this project

This project is a project in [LAION](https://laion.ai/) that aims at learning better audio understanding and getting more audio data. 
This is an opensource project. We adopt the codebase of [open_clip](https://github.com/mlfoundations/open_clip) for this project. 

many thanks to <a href="https://github.com/cfoster0/CLAP">@cfoster0</a> for allowing us to use his repo name.

## Architecture
Contrastive Language-Audio Pretraining, known as CLAP. Referring to the CLIP (Contrastive Language-Image Pretraining) architecture, the CLAP architecture is as follows.  
<p align="center">
  <img src="https://raw.githubusercontent.com/LAION-AI/CLAP/main/assets/audioclip-arch.png" alt="The Contrastive Language-Audio Pretraining Model Architecture" width="60%"/>
</p>

## Quick Start 
We provide the PyPI library for our CLAP model:
```bash
pip install laion-clap
```

For the documentation of the API, please refer to [hook.py](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/hook.py).

```python
import torch
import torchaudio
import laion_clap

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt('/path/to/checkpoint.safetensors')  # load from a local safetensors checkpoint

# Directly get audio embeddings from audio files
audio_file = [
    '/home/data/test_clap_short.wav',
    '/home/data/test_clap_long.wav'
]
audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
print(audio_embed[:,-20:])
print(audio_embed.shape)

# Get audio embeddings from audio data (torch tensors)
audio_data, sr = torchaudio.load('/home/data/test_clap_short.wav')
if sr != 48000:
    audio_data = torchaudio.transforms.Resample(sr, 48000)(audio_data)
audio_data = audio_data.mean(dim=0)  # Convert to mono
audio_data = audio_data.unsqueeze(0)  # Make it (1,T) or (N,T)
audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
print(audio_embed[:,-20:])
print(audio_embed.shape)

# Get text embeddings from texts:
text_data = ["I love the contrastive learning", "I love the pretrain model"] 
text_embed = model.get_text_embedding(text_data, use_tensor=True)
print(text_embed)
print(text_embed.shape)
```

## Pretrained Models
The pretrained checkpoints can be found in [here](https://huggingface.co/lukewys/laion_clap/tree/main).
Please refer to the previous section for how to load and run the checkpoints.
For the PyPI library, [630k-audioset-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt) and [630k-audioset-fusion-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-fusion-best.pt) are our default models (non-fusion and fusion)

We further provide below pretrained models according to your usages:

* For general audio less than 10-sec: [630k-audioset-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt) or [630k-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-best.pt)
* For general audio with variable-length: [630k-audioset-fusion-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-fusion-best.pt) or [630k-fusion-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-fusion-best.pt)
* For music: [music_audioset_epoch_15_esc_90.14.pt](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt)
* For music and speech: [music_speech_epoch_15_esc_89.25.pt](https://huggingface.co/lukewys/laion_clap/blob/main/music_speech_epoch_15_esc_89.25.pt)
* For speech, music and general audio: [music_speech_audioset_epoch_15_esc_89.98.pt](https://huggingface.co/lukewys/laion_clap/blob/main/music_speech_audioset_epoch_15_esc_89.98.pt)

The checkpoints list here for each model setting is the one with the highest average mAP score in training.
The average mAP score is calculated by averaging 4 scores: A-->T mAP@10 on AudioCaps, and T-->A mAP@10 on AudioCaps, A-->T mAP@10 on Clotho, and T-->A mAP@10 on Clotho.

To use above pretrained models, you need to load the ckpt by yourself, as:

Update 2023.4.7: we have released 3 larger CLAP models trained on music, speech dataset in addition to LAION-Audio-630k. Here are descriptions of the model and their performance:

 - `music_speech_audioset_epoch_15_esc_89.98.pt`: trained on music + speech + Audioset + LAION-Audio-630k. The zeroshot ESC50 performance is 89.98%, the GTZAN performance is 51%.
 - `music_audioset_epoch_15_esc_90.14.pt`: trained on music + Audioset + LAION-Audio-630k. The zeroshot ESC50 performance is 90.14%, the GTZAN performance is 71%.
 - `music_speech_epoch_15_esc_89.25.pt`: trained on music + speech + LAION-Audio-630k. The zeroshot ESC50 performance is 89.25%, the GTZAN performance is 69%.

The model uses a larger audio encoder. To load the model using the pip API:
```python
import laion_clap
model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
model.load_ckpt('checkpoint_path/checkpoint_name.pt')
```

Please note that this is a temporary release for people who are working on larger-scale down-stream task. 
We will release a more comprehensive version of the model with detailed experiments in the future.
Please take your own risk when using this model.

* All the new checkpoints did not trained with fusion. The training dataset size for `music_speech_audioset_epoch_15_esc_89.98.pt` is around 4M samples. The zeroshot GTZAN score is evaluated using the prompt `This audio is a <genre> song.`

<!-- We provide the CLAP's performance on audio classification tasks under the zero-shot setting or the supervised setting. More results can be found at our paper.
<p align="center">
  <img src="https://raw.githubusercontent.com/LAION-AI/CLAP/main/assets/clap-zeroshot.PNG" alt="Zero-shot Performance" width="100%"/>
</p> -->




## Environment Installation
If you want to check and reuse our model into your project instead of directly using the pip library, you need to install the same environment as we use, please run the following command:
```bash
conda create env -n clap python=3.10
conda activate clap
git clone https://github.com/LAION-AI/CLAP.git
cd CLAP
pip install -e .
```

## Core Code
Please refer to [hook.py](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/hook.py) and [model.py](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/model.py) to quickly get familiar with our model.

## Citation
If you find this project and the LAION-Audio-630K dataset useful, please cite our paper:
```
@inproceedings{laionclap2023,
  title = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author = {Wu*, Yusong and Chen*, Ke and Zhang*, Tianyu and Hui*, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2023}
}
@inproceedings{htsatke2022,
  author = {Ke Chen and Xingjian Du and Bilei Zhu and Zejun Ma and Taylor Berg-Kirkpatrick and Shlomo Dubnov},
  title = {HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2022}
}
```

## Acknowledgements

This project is working in progress, thus the codebase and model might not be perfect or bug-free. 
We will very much appreciate any kind of contribution or and issue raised.
If you find a bug or have any suggestion, please feel free to open an issue or contact us.
If you would actively contribute to this project, please join the discord of LAION.
