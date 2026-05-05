import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction="mean", ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
        self.ignore = ignore

    def forward(self, inputs, targets):
        """
        inputs: [N, C] (Logits, NOT probabilities)
        targets: [N] (Class indices)
        """
        # 1. Convert logits to probabilities
        prob = F.softmax(inputs, dim=1)

        # 2. Handle ignored labels
        if self.ignore is not None:
            mask = targets != self.ignore
            prob = prob[mask]
            targets = targets[mask]

        n_points, n_classes = prob.size()
        if n_points == 0:
            return prob.sum() * 0.0  # Handle empty batch

        losses = []
        # Iterate over all classes to ensure consistency
        for c in range(n_classes):
            target_c = (targets == c).float()
            input_c = prob[:, c]

            # Error for this class
            loss_c = (target_c - input_c).abs()

            # Sort errors in descending order
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]

            # Apply Lovasz extension gradient
            grad = lovasz_grad(target_c_sorted)
            losses.append(torch.dot(loss_c_sorted, grad))

        losses = torch.stack(losses)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses
