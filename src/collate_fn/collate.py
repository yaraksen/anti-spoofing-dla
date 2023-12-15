import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch import tensor

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch["text"] = [rec["text"] for rec in dataset_items]
    result_batch["audio_path"] = [rec["audio_path"] for rec in dataset_items]

    result_batch["audio"] = pad_sequence(
        [rec["audio"].squeeze(0) for rec in dataset_items], True, 0
    )
    # shape: batch_size_dim, feature_length_dim, time_dim = batch["spectrogram"].shape
    result_batch["spectrogram"] = (
        pad_sequence(
            [rec["spectrogram"].permute(2, 1, 0) for rec in dataset_items], True, 0
        )
        .squeeze(-1)
        .permute(0, 2, 1)
    )
    result_batch["spectrogram_length"] = tensor(
        [rec["spectrogram"].size(2) for rec in dataset_items]
    )
    result_batch["text_encoded"] = pad_sequence(
        [rec["text_encoded"].squeeze(0) for rec in dataset_items], True, 0
    )
    result_batch["text_encoded_length"] = tensor(
        [rec["text_encoded"].size(1) for rec in dataset_items]
    )
    return result_batch
