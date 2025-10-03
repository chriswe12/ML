import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# =====================
# Parameters (easy to adjust)
# =====================
params = {
    'batch_size': 128,
    'epochs': 10,
    'learning_rate': 1e3,
    'latent_dim': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# =====================
# Model Definition (VAE)
# =====================
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# =====================
# Data Loading
# =====================
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

# =====================
# Loss Function
# =====================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# =====================
# Training
# =====================
def train(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            running_loss += loss.item()
    return running_loss / len(loader.dataset)

# =====================
# Visualization
# =====================
def save_reconstructions(model, loader, device, epoch, num_images=8, folder='vae_reconstructions'):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:num_images].to(device)
    with torch.no_grad():
        recon, _, _ = model(imgs)
    imgs = imgs.cpu()
    recon = recon.cpu()
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
    for i in range(num_images):
        axes[0, i].imshow(imgs[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    os.makedirs(folder, exist_ok=True)
    img_path = os.path.join(folder, f'epoch_{epoch+1}.png')
    plt.savefig(img_path)
    plt.close(fig)

# =====================
# Main
# =====================
model = VAE(params['latent_dim']).to(params['device'])
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

train_losses = []
test_losses = []
recon_folder = 'vae_reconstructions'
# Remove old images before each run
if os.path.exists(recon_folder):
    for f in os.listdir(recon_folder):
        if f.endswith('.png'):
            os.remove(os.path.join(recon_folder, f))
else:
    os.makedirs(recon_folder)

for epoch in range(params['epochs']):
    train_loss = train(model, train_loader, optimizer, params['device'])
    test_loss = test(model, test_loader, params['device'])
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}/{params['epochs']} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
    save_reconstructions(model, test_loader, params['device'], epoch, folder=recon_folder)

# Save model weights after training
torch.save(model.state_dict(), 'vae_mnist_weights.pth')
