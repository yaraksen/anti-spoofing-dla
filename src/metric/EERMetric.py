from src.base.base_metric import BaseMetric
from src.metric.calculate_eer import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits, target, **kwargs):
        bonafide_logits = logits.detach().cpu().numpy()[..., 1]
        target = target.detach().cpu().numpy()
        mask_bonafide = target == 1
        return compute_eer(
            bonafide_logits[mask_bonafide], bonafide_logits[~mask_bonafide]
        )[0]
