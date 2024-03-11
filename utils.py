import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustAsymmetricLoss(nn.Module):
    """
    Computes the Robust Asymmetric Loss (RAL) for multiclass classification with class-specific weights.

    The RAL loss function extends the asymmetric loss by introducing class-specific weights to address class imbalance.

    Mathematical Formulation:
    L_RAL = ∑_{i=1}^{K} w_i * ((1 - y_i)^{\gamma_+} * log(\hat{y}_i) + psi(\hat{y}_i) * (1 - \hat{y}_i)^{\gamma_-} * log(1 - \hat{y}_i))

    Args:
        weights (torch.Tensor or numpy.ndarray): Class-specific weights of shape (num_classes,) or (num_classes, 1).
        gamma_pos (float): The positive focusing parameter (default: 2.0).
        gamma_neg (float): The negative focusing parameter (default: 1.0).
        lambda_hill (float): The hyperparameter for the Hill loss (default: 1.5).

    Shape:
        - Input: (BATCH_SIZE, n_classes) logits
        - Target: (BATCH_SIZE,) class labels (integer values)
        - Output: scalar loss

    Example:
        >>> num_classes = 10
        >>> weights = torch.ones(num_classes)  # Equal weights for all classes
        >>> criterion = RobustAsymmetricLoss(weights)
        >>> logits = torch.randn(32, num_classes)  # Example logits of shape (BATCH_SIZE, n_classes)
        >>> targets = torch.randint(0, num_classes, (32,))  # Example targets of shape (BATCH_SIZE,)
        >>> loss = criterion(logits, targets)
    """

    def __init__(self, weights, gamma_pos=2.0, gamma_neg=1.0, lambda_hill=1.5):
        super(RobustAsymmetricLoss, self).__init__()
        self.weights = torch.tensor(weights).to(torch.float32)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.lambda_hill = lambda_hill

    def forward(self, logits, targets):
        """
        Compute the Robust Asymmetric Loss (RAL) for multiclass classification.

        Args:
            logits (torch.Tensor): The predicted logits from the model of shape (BATCH_SIZE, n_classes).
            targets (torch.Tensor): The true class labels of shape (BATCH_SIZE,).

        Returns:
            torch.Tensor: The computed RAL loss.
        """
        # Ensure weights and logits are on the same device
        device = logits.device
        weights = self.weights.to(device)

        # Convert targets to one-hot encoding
        targets_one_hot = (
            F.one_hot(targets, num_classes=logits.size(1)).float().to(device)
        )

        # Compute predicted probabilities
        predicted_probs = F.softmax(logits, dim=1)

        # Compute Hill loss term
        hill_loss = self.lambda_hill - targets_one_hot * predicted_probs

        # Compute positive loss term
        positive_loss = (1 - targets_one_hot) ** self.gamma_pos * torch.log(
            predicted_probs
        )

        # Compute negative loss term with Hill regularization
        negative_loss = (
            hill_loss
            * (1 - predicted_probs) ** self.gamma_neg
            * torch.log(predicted_probs)
        )

        # Compute weighted loss
        weighted_loss = weights.unsqueeze(0) * (positive_loss + negative_loss)

        # Compute RAL loss
        ral_loss = -torch.sum(weighted_loss, dim=1)

        return ral_loss.mean()


class WeightedFocalLoss(nn.Module):
    """
    Computes the Weighted Focal Loss for multiclass classification with class-specific weights.

    The Weighted Focal Loss extends the Focal Loss by introducing class-specific weights to address class imbalance.

    Mathematical Formulation:
    L_Focal = -∑_{i=1}^{K} w_i * ((1 - y_i)^{\gamma} * log(\hat{y}_i) + (\hat{y}_i)^{\gamma} * (1 - y_i) * log(1 - \hat{y}_i))

    Args:
        weights (torch.Tensor or numpy.ndarray): Class-specific weights of shape (num_classes,) or (num_classes, 1).
        gamma (float): The focusing parameter (default: 2.0).
        epsilon (float): A small value to avoid numerical instability (default: 1e-9).

    Shape:
        - Input: (BATCH_SIZE, n_classes) logits
        - Target: (BATCH_SIZE,) class labels (integer values)
        - Output: scalar loss

    Example:
        >>> num_classes = 10
        >>> weights = torch.ones(num_classes)  # Equal weights for all classes
        >>> criterion = WeightedFocalLoss(weights)
        >>> logits = torch.randn(32, num_classes)  # Example logits of shape (BATCH_SIZE, n_classes)
        >>> targets = torch.randint(0, num_classes, (32,))  # Example targets of shape (BATCH_SIZE,)
        >>> loss = criterion(logits, targets)
    """

    def __init__(self, weights, gamma=2.0, epsilon=1e-7):
        super(WeightedFocalLoss, self).__init__()
        self.weights = torch.tensor(weights).to(torch.float32)
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        Compute the Weighted Focal Loss for multiclass classification.

        Args:
            logits (torch.Tensor): The predicted logits from the model of shape (BATCH_SIZE, n_classes).
            targets (torch.Tensor): The true class labels of shape (BATCH_SIZE,).

        Returns:
            torch.Tensor: The computed Weighted Focal Loss.
        """
        # Ensure weights and logits are on the same device
        device = logits.device
        weights = self.weights.to(device)

        # Convert targets to one-hot encoding
        targets_one_hot = (
            F.one_hot(targets, num_classes=logits.size(1)).float().to(device)
        )

        # Compute predicted probabilities with epsilon clipping
        predicted_probs = F.softmax(logits, dim=1)
        predicted_probs_clipped = torch.clamp(
            predicted_probs, self.epsilon, 1.0 - self.epsilon
        )

        # Compute focal loss term for correctly classified examples
        focal_loss_correct = (
            (1 - predicted_probs_clipped) ** self.gamma
            * targets_one_hot
            * torch.log(predicted_probs_clipped)
        )

        # Compute focal loss term for incorrectly classified examples
        focal_loss_incorrect = (
            predicted_probs_clipped**self.gamma
            * (1 - targets_one_hot)
            * torch.log(1 - predicted_probs_clipped)
        )

        # Compute weighted loss
        weighted_loss = weights.unsqueeze(0) * (
            focal_loss_correct + focal_loss_incorrect
        )

        # Compute Weighted Focal Loss
        focal_loss = -torch.sum(weighted_loss, dim=1)
        #         print(predicted_probs, predicted_probs_clipped, focal_loss_correct, focal_loss_incorrect, focal_loss)
        return focal_loss.mean()
