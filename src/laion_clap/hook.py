"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: https://arxiv.org/abs/2211.06687
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""
import torch
import torchaudio
from clap_module import create_model
from .training.data import get_audio_features
from .training.data import int16_to_float32_torch, float32_to_int16_torch

from transformers import RobertaTokenizer
from clap_module.factory import load_state_dict


class CLAP_Module(torch.nn.Module):
    def __init__(self, enable_fusion=False, device=None, amodel= 'HTSAT-tiny', tmodel='roberta') -> None:
        """Initialize CLAP Model

        Parameters
        ----------
        enable_fusion: bool
            if true, it will create the fusion clap model, otherwise non-fusion clap model (default: false) 
        device: str
            if None, it will automatically detect the device (gpu or cpu)
        amodel: str
            audio encoder architecture, default: HTSAT-tiny
        tmodel: str
            text encoder architecture, default: roberta
        """
        super(CLAP_Module, self).__init__()
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        precision = 'fp32'

        if enable_fusion:
            fusion_type = 'aff_2d'
            model, model_cfg = create_model(
                amodel,
                tmodel,
                precision=precision,
                device=device,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type
            )
        else:
            model, model_cfg = create_model(
                amodel,
                tmodel,
                precision=precision,
                device=device,
                enable_fusion=enable_fusion
            )
        self.enable_fusion = enable_fusion
        self.model = model
        self.model_cfg = model_cfg
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return result

    def load_ckpt(self, ckpt: str):
        """Load the pretrained checkpoint of CLAP model

        Parameters
        ----------
        ckpt: str
            Path to the safetensors checkpoint file to load.
            Checkpoints can be converted from the original .pt format using:
            `from safetensors.torch import save_file; save_file(torch.load('model.pt'), 'model.safetensors')`
        """
        if ckpt is None:
            raise ValueError(
                "Checkpoint path must be specified. Auto-download is no longer supported. "
                "Please provide a path to a local safetensors checkpoint file."
            )
        state_dict = load_state_dict(ckpt, skip_params=True)
        self.model.load_state_dict(state_dict)
    
    def get_audio_embedding_from_filelist(self, x, use_tensor=False):
        """get audio embeddings from the audio file list

        Parameters
        ----------
        x: List[str] (N,): 
            an audio file list to extract features, audio files can have different lengths (as we have the feature fusion machanism)
        use_tensor: boolean:
            if True, returns a tensor attached to the computation graph (default: False).
            If False, returns a detached tensor on CPU.
        Returns
        ----------
        audio_embed : torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """ 
        self.model.eval()
        audio_input = []
        for f in x:
            # load the waveform of the shape (T,), should resample to 48000
            audio_waveform, sr = torchaudio.load(f)
            # Convert to mono if stereo
            if audio_waveform.shape[0] > 1:
                audio_waveform = audio_waveform.mean(dim=0, keepdim=True)
            audio_waveform = audio_waveform.squeeze(0)
            # Resample to 48000 if needed
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
                audio_waveform = resampler(audio_waveform)
            # quantize
            audio_waveform = int16_to_float32_torch(float32_to_int16_torch(audio_waveform))
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000, 
                data_truncating='fusion' if self.enable_fusion else 'rand_trunc', 
                data_filling='repeatpad',
                audio_cfg=self.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)
        audio_embed = self.model.get_audio_embedding(audio_input)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu()
        return audio_embed


    def get_audio_embedding_from_data(self, x, use_tensor=False):
        """get audio embeddings from the audio data

        Parameters
        ----------
        x: torch.Tensor (N,T): 
            audio data, must be mono audio tracks.
        use_tensor: boolean:
            if True, x should be the tensor input and the output will be the tensor, preserving the gradient (default: False).      
            Note that if 'use tensor' is set to True, it will not do the quantize of the audio waveform (otherwise the gradient will not be preserved).
        Returns
        ----------
        audio embed: torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """ 
        self.model.eval()
        audio_input = []
        for audio_waveform in x:          
            # quantize
            if not use_tensor:
                audio_waveform = int16_to_float32_torch(float32_to_int16_torch(audio_waveform))
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000, 
                data_truncating='fusion' if self.enable_fusion else 'rand_trunc', 
                data_filling='repeatpad',
                audio_cfg=self.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)
        audio_embed = self.model.get_audio_embedding(audio_input)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu()
        return audio_embed

    def get_text_embedding(self, x, tokenizer = None, use_tensor = False):
        """get text embeddings from texts

        Parameters
        ----------
        x: List[str] (N,): 
            text list 
        tokenizer: func:
            the tokenizer function, if not provided (None), will use the default Roberta tokenizer.
        use_tensor: boolean:
            if True, returns a tensor attached to the computation graph (default: False).
            If False, returns a detached tensor on CPU.
        Returns
        ----------
        text_embed : torch.Tensor (N,D):
            text embeddings that extracted from texts
        """ 
        self.model.eval()
        if tokenizer is not None:
            text_input = tokenizer(x)
        else:
            text_input = self.tokenizer(x)
        text_embed = self.model.get_text_embedding(text_input)
        if not use_tensor:
            text_embed = text_embed.detach().cpu()
        return text_embed
        
    
