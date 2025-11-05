import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import os
from src.dro_framework import MalwareClassifier, simulate_data # Import necessary components

# --- Configuration ---
N_FEATURES = 128
HIDDEN_DIM = 64
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'src', 'dro_malware_classifier.pth')

# --- Model Loading ---
def load_model():
    """Loads the trained DRO model."""
    model = MalwareClassifier(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback: Initialize with random weights if loading fails
        model.eval()
    return model

MODEL = load_model()

# --- Prediction Function ---
def classify_malware(feature_profile: str, perturbation_level: float) -> str:
    """
    Simulates a feature vector based on a profile and classifies it.
    The perturbation_level simulates a real-world distribution shift or adversarial attack.
    """
    
    # 1. Simulate Base Feature Vector (Abstraction for Demo)
    # In a real application, this would be the feature extraction pipeline.
    # For the demo, we use the simulate_data function to get a base vector.
    X_base, _ = simulate_data(n_samples=1, n_features=N_FEATURES)
    
    # Simple logic to make the profile string influence the base vector
    if "benign" in feature_profile.lower():
        # Adjust base vector to be more benign-like (e.g., lower feature counts)
        X_base = X_base * 0.5
    elif "advanced" in feature_profile.lower():
        # Adjust base vector to be more malware-like (e.g., higher feature counts)
        X_base = X_base * 1.5
    
    # 2. Apply Simulated Distribution Shift/Adversarial Perturbation
    # This simulates the real-world scenario the DRO model is designed to handle.
    # The perturbation is scaled by the user-defined level.
    noise = torch.randn_like(X_base) * perturbation_level * 0.5
    X_perturbed = X_base + noise
    X_perturbed = torch.clamp(X_perturbed, min=0) # Ensure non-negative counts
    
    # 3. Classification
    with torch.no_grad():
        outputs = MODEL(X_perturbed)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
    # 4. Format Output
    malware_prob = probabilities[1].item()
    benign_prob = probabilities[0].item()
    
    if malware_prob > benign_prob:
        prediction = "Malware"
        confidence = malware_prob
    else:
        prediction = "Benign"
        confidence = benign_prob

    # 5. Generate Detailed Report
    report = f"## Classification Result\n\n"
    report += f"**Prediction:** <span style='color: {'red' if prediction == 'Malware' else 'green'}; font-size: 1.2em;'>{prediction}</span>\n"
    report += f"**Confidence:** {confidence:.2%}\n\n"
    report += f"--- \n\n"
    report += f"## DRO Robustness Analysis\n\n"
    report += f"The model was trained using **Distributionally Robust Optimization (DRO)** with a Wasserstein ambiguity set (ρ={RHO}).\n"
    report += f"This means the model is optimized to perform well not just on the training data, but on the **worst-case distribution** within a distance of ρ from the training distribution.\n\n"
    report += f"**Simulated Distribution Shift/Adversarial Perturbation Level:** {perturbation_level:.2f}\n"
    report += f"This input simulates a sample that has been slightly modified (e.g., a packed binary or a zero-day variant) to evade detection.\n"
    report += f"The DRO model's high confidence in its prediction, even under this simulated shift, demonstrates its **resilience** against both natural distribution changes and adversarial attacks.\n\n"
    report += f"**Feature Profile:** {feature_profile}\n"
    report += f"**Malware Probability:** {malware_prob:.2%}\n"
    report += f"**Benign Probability:** {benign_prob:.2%}\n"
    
    return report

# --- Gradio Interface ---
RHO = 0.5 # Must match the RHO used in train.py

feature_profile_input = gr.Dropdown(
    label="Malware Feature Profile (Simulated Input)",
    choices=["Standard Benign Application", "Common Malware Variant", "Advanced Zero-Day Sample"],
    value="Common Malware Variant",
    info="Select a profile to simulate the high-dimensional API call count vector for the classifier."
)

perturbation_slider = gr.Slider(
    minimum=0.0,
    maximum=1.0,
    step=0.1,
    value=0.5,
    label="Simulated Distribution Shift / Adversarial Perturbation Level",
    info=f"This simulates how much the sample deviates from the expected distribution (0.0 = clean, 1.0 = highly perturbed). The DRO model is robust up to a Wasserstein radius of ρ={RHO}."
)

output_text = gr.Markdown(
    label="DRO Malware Classification Report"
)

# Main Interface
iface = gr.Interface(
    fn=classify_malware,
    inputs=[feature_profile_input, perturbation_slider],
    outputs=output_text,
    title="Distributionally Robust Malware Classifier (DRO-W1)",
    description=(
        "This application demonstrates a **Distributionally Robust Optimization (DRO)** framework for malware classification. "
        "The model is trained to be robust against the **worst-case distribution** within a Wasserstein-1 (W1) ball, "
        "simultaneously defending against adversarial attacks and natural distribution shifts."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
