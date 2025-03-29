import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.cifar import get_cifar10_loader
from src.models.blocks import VQVAE
from src.utils.common import extract_tokens_from_vqvae
from torchvision import transforms
from tqdm import tqdm

# Configurations
num_epochs = 20
n_hidden = 256
n_residual_hidden = 128
n_residual_layers = 3
n_embeddings = 2048
embedding_dim = 64
beta = 0.25
learning_rate = 3e-4
log_interval = 100
batch_size = 350
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
trainloader, testloader = get_cifar10_loader(batch_size=1)

# Load model
model = VQVAE(
    n_hidden, n_residual_hidden, n_residual_layers, n_embeddings, embedding_dim, beta
).to(device)
model.load_state_dict(
    torch.load("best_models/v5/vqvae_model_aircraft_v5_loss_0.0053.pth")
)
model.eval()

if __name__ == "__main__":
    token_list = extract_tokens_from_vqvae(model, trainloader, device)
    np.save("Tokens.npy", token_list)
