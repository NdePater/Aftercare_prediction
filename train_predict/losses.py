import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import one_hot

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification tasks.
    
    Attributes:
        alpha (float): Weighting factor for the positive class.
        gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
        eps (float): Small value to avoid division by zero.
        bce_with_logits (torch.nn.Module): Binary Cross Entropy with Logits loss function.
    """
    def __init__(self, alpha=None, gamma=0, eps=1e-7):
        """
        Initializes the FocalLoss class with the given parameters.
        
        Args:
            alpha (float, optional): Weighting factor for the positive class. Defaults to None.
            gamma (float, optional): Focusing parameter to reduce the relative loss for well-classified examples. Defaults to 0.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.
        """
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        """
        Calculates the focal loss between the inputs and targets.
        
        Args:
            inputs (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Calculated focal loss.
        """
        # from pytorch
        inputs = inputs.squeeze()
        inputs, targets = inputs.float(), targets.float()

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.sum()