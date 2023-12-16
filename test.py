import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_model
from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser
import torchaudio
import torch.nn.functional as F 

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"
        

def main(config, audio_path: str, out_file: str):
    logger = config.get_logger("test")
    audio_path = Path(audio_path)

    # define cpu or gpu if possible
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(device)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        with open(out_file, "w") as out_file:
            for file in tqdm(audio_path.iterdir(), desc=f"Processing..."):
                audio, sr = torchaudio.load(file)
                assert sr == 16000
                
                max_len = 64000
                if audio.shape[1] < max_len:
                    repeat_value = max_len // audio.shape[1] + 1
                    audio = audio.repeat(1, repeat_value)[:, :max_len]
                else:
                    audio = audio[:, :max_len]
                
                spoof_proba = F.softmax(model(audio.unsqueeze(1).to(device))[0], dim=0)[0].item()
                out_file.write(f"Spoof probability for {file}: {spoof_proba:.7f}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )
    args.add_argument(
        "-ap",
        "--audio_path",
        default="test_audios",
        type=str,
        help="File with checkpoint of vocoder model",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    with open(args.config) as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    main(config, args.audio_path, args.output)
