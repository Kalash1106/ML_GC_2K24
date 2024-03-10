import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustAsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=1.0, lambda_hill=1.5):
        super(RobustAsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.lambda_hill = lambda_hill

    def forward(self, logits, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        # Compute predicted probabilities
        predicted_probs = F.softmax(logits, dim=1)

        # Compute positive loss term
        positive_loss = (1 - targets_one_hot) ** self.gamma_pos * torch.log(
            predicted_probs
        )
        positive_loss = torch.sum(positive_loss, dim=1)

        # Compute Hill loss term
        hill_loss = self.lambda_hill - targets_one_hot * predicted_probs

        # Compute negative loss term with Hill regularization
        negative_loss = (
            hill_loss
            * (1 - predicted_probs) ** self.gamma_neg
            * torch.log(predicted_probs)
        )
        negative_loss = torch.sum(negative_loss, dim=1)

        # Compute RAL loss
        ral_loss = positive_loss + negative_loss

        return -ral_loss.mean()
