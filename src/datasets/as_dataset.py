from pathlib import Path
from tqdm import tqdm
import torchaudio

from torch import randint
from torch.utils.data import Dataset


class ASVSpoofDataset(Dataset):
    def __init__(self, data_path: str, protocol_path: str):
        self.data_path = Path(data_path)

        # flac_path = data_path / "flac"
        # self.flac_files = list(flac_path.glob("**/*.flac"))
        self.file_path = []
        self.target = []

        with open(protocol_path, "r") as f:
            for file_info in f.readlines():
                # LA_0069 LA_D_1105538 - - bonafide
                file_info = file_info.strip().split()
                file_id, target = file_info[1], file_info[4]
                self.file_path.append(data_path / f"{file_id}.flac")
                self.target.append(int(target == "bonafide"))

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.wav_files[idx])
        return {"audio": wav, "target": self.mel_creator(wav.detach())}
