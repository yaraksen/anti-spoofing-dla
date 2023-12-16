from pathlib import Path
from tqdm import tqdm
import torchaudio

from torch import randint
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np


class ASVSpoofDataset(Dataset):
    def __init__(self, data_path: str, protocol_path: str, limit: int = None):
        self.data_path = Path(data_path) / "flac"
        self.file_paths = []
        self.targets = []

        with open(protocol_path, "r") as f:
            for file_info in f.readlines():
                # LA_0069 LA_D_1105538 - - bonafide
                file_info = file_info.strip().split()
                file_id, target = file_info[1], file_info[4]
                self.file_paths.append(self.data_path / f"{file_id}.flac")
                self.targets.append(int(target == "bonafide"))

        if limit is not None:
            perm = np.random.permutation(len(self.file_paths))[:limit]
            self.file_paths = np.array(self.file_paths)[perm]
            self.targets = np.array(self.targets)[perm]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        flac, _ = torchaudio.load(self.file_paths[idx])
        max_len = 64000
        if flac.shape[1] < max_len:
            # flac = F.pad(
            #     flac.unsqueeze(0), (0, max_len - flac.shape[1]), mode="circular"
            # ).squeeze(0) # Does not allow more than one circle
            repeat_value = max_len // flac.shape[1] + 1
            flac = flac.repeat(1, repeat_value)[:, :max_len]
        else:
            flac = flac[:, :max_len]
        return {"audio": flac, "target": self.targets[idx]}
