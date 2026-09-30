from pathlib import Path

import numpy as np
import torch

from src.checkpoint_io import read_json
from src.token_memory_edit import FrozenSupCon, TokenDictionary, TokenMemoryPredictor


INTERVENTION_TYPE = "residual_component_ablation"


def load_predictor(fit_directory, device="cpu"):
    directory = Path(fit_directory)
    config = read_json(directory / "model_config.json")
    if config["intervention_type"] != INTERVENTION_TYPE:
        raise ValueError("Unknown component intervention")
    dictionary = TokenDictionary(config)
    with np.load(directory / "model.npz", allow_pickle=False) as data:
        dictionary.load_state_dict({key: torch.from_numpy(data[key].copy()) for key in data.files}, strict=True)
    with np.load(directory / "normalization.npz", allow_pickle=False) as data:
        mean, scale = data["mean"].copy(), data["scale"].copy()
    with np.load(directory / "gains.npz", allow_pickle=False) as data:
        gains, selected = data["gains"].copy(), data["selected_indices"].copy()
    strength = float(config["intervention_strength"])
    if not 0 <= strength <= 1 or gains.shape != (dictionary.latent_dim,):
        raise ValueError("Invalid component ablation dose or shape")
    if selected.ndim != 1 or selected.dtype.kind not in "iu" or len(np.unique(selected)) != len(selected):
        raise ValueError("Invalid selected component indices")
    if np.any(selected < 0) or np.any(selected >= dictionary.latent_dim):
        raise ValueError("Selected component index is outside the dictionary")
    expected = np.zeros(dictionary.latent_dim, dtype=np.float32)
    expected[selected] = -strength
    if not np.array_equal(gains, expected):
        raise ValueError("Component gains differ from the saved ablation set and dose")
    return TokenMemoryPredictor(dictionary, mean, scale, gains, FrozenSupCon(config["reference_fit"]), device)
