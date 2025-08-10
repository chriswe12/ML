# MNIST Autoencoder & Variational Autoencoder

This project contains scripts for training and exploring autoencoders and variational autoencoders (VAE) on the MNIST dataset using PyTorch.

## Files

- `autoencoder_mnist.py`: Trains a standard autoencoder on MNIST. Saves model weights and reconstruction images after each epoch.
- `vae_mnist.py`: Trains a variational autoencoder (VAE) on MNIST. Saves model weights and reconstruction images after each epoch.
- `interactive_latent_explorer.py`: Interactive tool to explore the latent space of a trained autoencoder. Adjust latent variables with sliders and see how the output image changes in real time.

## Usage

1. **Train an Autoencoder**
   ```bash
   python autoencoder_mnist.py
   ```
   - Model weights saved as `autoencoder_mnist_weights.pth`
   - Reconstructions saved in `reconstructions/`

2. **Train a Variational Autoencoder (VAE)**
   ```bash
   python vae_mnist.py
   ```
   - Model weights saved as `vae_mnist_weights.pth`
   - Reconstructions saved in `vae_reconstructions/`

3. **Explore Latent Space Interactively**
   ```bash
   python interactive_latent_explorer.py
   ```
   - Make sure you have trained the autoencoder and have `autoencoder_mnist_weights.pth` available.
   - Adjust latent variables with sliders to see how the output image changes.

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- matplotlib

Install dependencies with:
```bash
pip install torch torchvision matplotlib
```

## Customization
- You can easily adjust model parameters (batch size, epochs, latent dimension, etc.) at the top of each script.
- For VAE, you can add more advanced features or visualizations as needed.

## License
MIT
