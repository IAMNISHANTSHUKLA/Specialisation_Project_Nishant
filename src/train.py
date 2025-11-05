import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dro_framework import simulate_data, MalwareClassifier, dro_loss
import os

# Configuration
N_FEATURES = 128
HIDDEN_DIM = 64
N_SAMPLES = 5000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 10
RHO = 0.5  # Wasserstein-1 radius (robustness budget)
N_PERTURB_STEPS = 5 # PGD steps for inner maximization

def train_model():
    # 1. Data Preparation
    X, y = simulate_data(n_samples=N_SAMPLES, n_features=N_FEATURES)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model, Loss, and Optimizer
    model = MalwareClassifier(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print(f"Starting DRO training with RHO={RHO} for {N_EPOCHS} epochs...")
    for epoch in range(N_EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()

            # Calculate DRO Loss (Worst-Case Loss)
            loss = dro_loss(
                data, target, model, criterion, 
                rho=RHO, n_perturb_steps=N_PERTURB_STEPS
            )
            
            # Outer Minimization (Standard Backpropagation)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += target.size(0)

            # Calculate accuracy on the *original* data for monitoring
            model.eval()
            with torch.no_grad():
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == target).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_samples
                print(f"Epoch {epoch+1}/{N_EPOCHS}, Batch {batch_idx+1}/{len(dataloader)} | Loss: {avg_loss:.4f} | Acc (Clean): {accuracy:.4f}")

        avg_epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples
        print(f"--- Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f} | Total Acc (Clean): {epoch_accuracy:.4f} ---")

    # 4. Save Model
    model_path = os.path.join(os.path.dirname(__file__), 'dro_malware_classifier.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    # Ensure the current directory is added to the path for imports
    import sys
    sys.path.append(os.path.dirname(__file__))
    
    train_model()
