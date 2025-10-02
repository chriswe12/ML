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
    'learning_rate': 1e-3,
    'latent_dim': 25,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# =====================
# Model Definition
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
# Data Loading
# =====================
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

# =====================
# Training
# =====================
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def save_reconstructions(model, loader, device, epoch, num_images=18, folder='reconstructions'):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:num_images].to(device)
    with torch.no_grad():
        recons = model(imgs)
    imgs = imgs.cpu()
    recons = recons.cpu()
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
    for i in range(num_images):
        axes[0, i].imshow(imgs[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i].squeeze(), cmap='gray')
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
model = Autoencoder(params['latent_dim']).to(params['device'])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

train_losses = []
test_losses = []
recon_folder = 'reconstructions'
# Remove old images before each run
if os.path.exists(recon_folder):
    for f in os.listdir(recon_folder):
        if f.endswith('.png'):
            os.remove(os.path.join(recon_folder, f))
else:
    os.makedirs(recon_folder)

for epoch in range(params['epochs']):
    train_loss = train(model, train_loader, criterion, optimizer, params['device'])
    test_loss = test(model, test_loader, criterion, params['device'])
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}/{params['epochs']} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
    save_reconstructions(model, test_loader, params['device'], epoch, folder=recon_folder)

# Save model weights after training
torch.save(model.state_dict(), 'autoencoder_mnist_weights.pth')
