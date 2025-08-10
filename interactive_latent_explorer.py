import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from torchvision import datasets, transforms

# =====================
# Parameters
# =====================
LATENT_DIM = 5  # Change if needed to match your trained model
MODEL_WEIGHTS = 'autoencoder_mnist_weights.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================
# Model Definition (must match training script)
# =====================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# =====================
# Load Model and Data
# =====================
model = Autoencoder(LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Select a few images to play with
num_samples = 1
index = np.random.choice(len(test_dataset), 1)[0]
sample_img = test_dataset[index][0].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    sample_latent = model.encoder(sample_img)

# =====================
# Interactive Plot
# =====================
def interactive_latent_explorer(model, sample_imgs, sample_latents, device):
    # Calculate required figure height for sliders
    slider_height = 0.02
    slider_spacing = 0.01
    total_slider_height = num_samples * LATENT_DIM * (slider_height + slider_spacing)
    # Ensure bottom margin is less than 0.8 (top=1.0)
    bottom_margin = min(0.25 + total_slider_height, 0.8)
    fig_height = num_samples * 3 + total_slider_height * 10  # More space for images and sliders
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*3, fig_height))
    plt.subplots_adjust(bottom=bottom_margin)
    latent_values = sample_latent.cpu().numpy()[0]
    fig, ax = plt.subplots(1, 1, figsize=(3, 5))
    slider_height = 0.02
    slider_spacing = 0.01
    total_slider_height = LATENT_DIM * (slider_height + slider_spacing)
    bottom_margin = min(0.25 + total_slider_height, 0.8)
    plt.subplots_adjust(bottom=bottom_margin)

    # Initial reconstruction
    with torch.no_grad():
        recon = model.decoder(torch.tensor(latent_values, dtype=torch.float32, device=device).unsqueeze(0)).detach().cpu().numpy()[0]
    img_plot = ax.imshow(recon[0], cmap='gray', vmin=0, vmax=1)
    ax.set_title('Reconstructed Sample')
    ax.axis('off')

    # Sliders for each latent dimension
    sliders = []
    slider_axes = []
    for l in range(LATENT_DIM):
        slider_bottom = 0.25 + l * (slider_height + slider_spacing)
        ax_slider = plt.axes([0.1, slider_bottom, 0.8, slider_height])
        slider = Slider(ax_slider, f'Latent {l+1}', -3.0, 3.0, valinit=latent_values[l])
        sliders.append(slider)
        slider_axes.append(ax_slider)

    def update(val):
        new_latent = np.array([slider.val for slider in sliders])
        with torch.no_grad():
            new_recon = model.decoder(torch.tensor(new_latent, dtype=torch.float32, device=device).unsqueeze(0)).detach().cpu().numpy()[0]
        img_plot.set_data(new_recon[0])
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    # Button to reset sliders to original latent values
    resetax = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    def reset(event):
        for l, slider in enumerate(sliders):
            slider.set_val(latent_values[l])
    button.on_clicked(reset)

    plt.show()

interactive_latent_explorer(model, sample_img, sample_latent, DEVICE)
