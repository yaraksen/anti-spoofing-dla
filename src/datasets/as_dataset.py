from pathlib import Path
from tqdm import tqdm
import torchaudio

from torch import randint
from torch.utils.data import Dataset
import torch.nn.functional as F


class ASVSpoofDataset(Dataset):
    def __init__(self, data_path: str, protocol_path: str):
        self.data_path = Path(data_path) / "flac"

        # flac_path = data_path / "flac"
        # self.flac_files = list(flac_path.glob("**/*.flac"))
        self.file_paths = []
        self.targets = []

        with open(protocol_path, "r") as f:
            for file_info in f.readlines():
                # LA_0069 LA_D_1105538 - - bonafide
                file_info = file_info.strip().split()
                file_id, target = file_info[1], file_info[4]
                self.file_paths.append(self.data_path / f"{file_id}.flac")
                self.targets.append(int(target == "bonafide"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        flac, _ = torchaudio.load(self.file_paths[idx])

        assert flac.dim() == 2, f"{flac.dim()} is not 2"

        max_len = 64000
        if flac.shape[0] < max_len:
            # flac = F.pad(
            #     flac.unsqueeze(0), (0, max_len - flac.shape[0]), mode="circular"
            # ).squeeze(0)
            # TODO: use different padding method
            pass
        else:
            flac = flac[:, :max_len]

        assert flac.dim == 2
        assert flac.shape[0] == max_len

        return {"audio": flac, "target": self.targets[idx]}
