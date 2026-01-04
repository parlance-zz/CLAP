"""
Audio data processing utilities for CLAP model inference.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms
import torchaudio
from contextlib import suppress


def int16_to_float32_torch(x):
    """Convert int16 audio to float32."""
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x):
    """Convert float32 audio to int16."""
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)


def get_mel(audio_data, audio_cfg):
    """Get mel spectrogram from audio data.
    
    Args:
        audio_data: torch.Tensor of shape (T,) containing audio samples.
        audio_cfg: dict with audio configuration (sample_rate, window_size, hop_size, mel_bins, fmin, fmax).
    
    Returns:
        mel: torch.Tensor of shape (T, n_mels) containing log mel spectrogram.
    """
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=audio_cfg['mel_bins'],
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    ).to(audio_data.device)
    
    mel = mel_tf(audio_data)
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)


def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg, require_grad=False):
    """
    Calculate and add audio features to sample.
    
    Args:
        sample: a dict containing all the data of current sample.
        audio_data: a tensor of shape (T) containing audio data.
        max_len: the maximum length of audio data.
        data_truncating: the method of truncating data ('rand_trunc' or 'fusion').
        data_filling: the method of filling data ('repeatpad', 'pad', or 'repeat').
        audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
        require_grad: whether to require gradient for audio data.
    
    Returns:
        sample: dict with added 'longer', 'waveform', and optionally 'mel_fusion' keys.
    """
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data, audio_cfg)
                # split to three parts
                chunk_frames = max_len // audio_cfg['hop_size'] + 1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    # Split range [0, total_frames - chunk_frames] into 3 parts using torch
                    range_len = total_frames - chunk_frames + 1
                    ranges = torch.tensor_split(torch.arange(range_len), 3)
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges = (ranges[0], torch.tensor([0]), ranges[2])
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges = (ranges[0], ranges[1], torch.tensor([0]))
                    # randomly choose index for each part
                    idx_front = ranges[0][torch.randint(len(ranges[0]), (1,))].item()
                    idx_middle = ranges[1][torch.randint(len(ranges[1]), (1,))].item()
                    idx_back = ranges[2][torch.randint(len(ranges[2]), (1,))].item()
                    # select mel
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                    # shrink the mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, audio_cfg['mel_bins']])(mel[None])[0]

                    # stack
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = torch.randint(0, overflow + 1, (1,)).item()
            audio_data = audio_data[idx: idx + max_len]

        else:  # padding if too short
            if len(audio_data) < max_len:  # do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample
