import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification

# 1. Data Simulation
def simulate_data(n_samples=1000, n_features=128, n_informative=32, n_classes=2):
    """
    Simulates a high-dimensional, discrete malware dataset.
    Returns X (counts) and y (labels).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.01,
        random_state=42
    )
    # Convert to non-negative integer counts
    X = (np.abs(X) * 100).astype(np.int32)
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()

# 2. Classifier Model
class MalwareClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=2):
        super(MalwareClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 3. DRO Loss Function (Wasserstein-1)
def dro_loss(x, y, model, criterion, rho, n_perturb_steps=10, step_size=0.01):
    """
    Calculates the distributionally robust loss for a batch.
    This finds the worst-case perturbation within the W-1 ball and computes the loss.
    """
    model.eval() # Freeze model parameters for perturbation search

    # Create a copy of the input to be perturbed
    x_adv = x.clone().detach().requires_grad_(True)

    # Find the adversarial perturbation via PGD-like steps
    for _ in range(n_perturb_steps):
        with torch.enable_grad():
            loss_adv = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss_adv, [x_adv])[0]
        
        # Project the perturbation back into the L1 ball of radius rho
        # For discrete counts, this is a simplified projection.
        x_adv = x_adv + step_size * grad.sign()
        perturbation = x_adv - x
        # Projection step for L1 norm
        perturbation = torch.clamp(perturbation, -rho, rho)
        x_adv = x + perturbation
        # Ensure features remain non-negative
        x_adv = torch.clamp(x_adv, min=0)

    model.train() # Unfreeze model for training

    # Calculate the loss on the worst-case perturbed data
    worst_case_loss = criterion(model(x_adv), y)
    return worst_case_loss
