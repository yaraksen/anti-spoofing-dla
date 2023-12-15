import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, stack

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch["audio"] = stack([rec["audio"] for rec in dataset_items], dim=0)
    result_batch["target"] = tensor([rec["target"] for rec in dataset_items])
    return result_batch
