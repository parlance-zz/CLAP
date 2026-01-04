import json
import re
from copy import deepcopy
from pathlib import Path
from packaging import version

import torch
import transformers
from safetensors.torch import load_file as load_safetensors

from .model import CLAP, convert_weights_to_fp16

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "audio_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    """Load model state dict from a safetensors checkpoint file.
    
    Parameters
    ----------
    checkpoint_path: str
        Path to the safetensors checkpoint file.
    map_location: str
        Device to load the checkpoint to. Default is "cpu".
    skip_params: bool
        If True, removes "module." prefix from keys and handles transformers compatibility.
    
    Returns
    -------
    state_dict: dict
        The model state dict.
    """
    state_dict = load_safetensors(checkpoint_path, device=map_location)
    
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # removing position_ids to maintain compatibility with latest transformers update        
        if version.parse(transformers.__version__) >= version.parse("4.31.0") and "text_branch.embeddings.position_ids" in state_dict:
            del state_dict["text_branch.embeddings.position_ids"]
    return state_dict


def create_model(
    amodel_name: str,
    tmodel_name: str,
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    enable_fusion: bool = False,
    fusion_type: str = 'None'
):
    """Create a CLAP model.
    
    Parameters
    ----------
    amodel_name: str
        Audio model architecture name (e.g., 'HTSAT-tiny', 'HTSAT-base').
    tmodel_name: str
        Text model architecture name (e.g., 'roberta', 'bert').
    precision: str
        Model precision ('fp32' or 'fp16'). Default is 'fp32'.
    device: torch.device
        Device to load the model to.
    jit: bool
        If True, compile the model with torch.jit.script.
    force_quick_gelu: bool
        If True, use QuickGELU instead of native GELU.
    enable_fusion: bool
        If True, enable audio fusion for variable-length audio.
    fusion_type: str
        Type of fusion to use (e.g., 'aff_2d').
    
    Returns
    -------
    model: CLAP
        The CLAP model.
    model_cfg: dict
        The model configuration.
    """
    amodel_name = amodel_name.replace("/", "-")
    
    if amodel_name not in _MODEL_CONFIGS:
        raise RuntimeError(f"Model config for {amodel_name} not found. Available models: {list_models()}")
    
    model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])

    if force_quick_gelu:
        model_cfg["quick_gelu"] = True

    model_cfg["text_cfg"]["model_type"] = tmodel_name
    model_cfg["enable_fusion"] = enable_fusion
    model_cfg["fusion_type"] = fusion_type
    model = CLAP(**model_cfg)
        
    model.to(device=device)
    if precision == "fp16":
        assert device.type != "cpu"
        convert_weights_to_fp16(model)

    if jit:
        model = torch.jit.script(model)

    return model, model_cfg


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()
