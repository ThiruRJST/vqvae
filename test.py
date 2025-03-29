import mlflow.pytorch
import numpy as np
import torch

from PIL import Image
from src.data.cifar import get_cifar10_loader
from src.models.blocks import VQVAE
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
trainloader, testloader = get_cifar10_loader(batch_size)

# Load model
model = VQVAE(
    n_hidden, n_residual_hidden, n_residual_layers, n_embeddings, embedding_dim, beta
).to(device)
model.load_state_dict(torch.load("best_models/v5/vqvae_model_aircraft_v5_loss_0.0053.pth"))
model.eval()

pbar = tqdm(enumerate(testloader), total=len(testloader))

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

for i, (x, _) in pbar:
    x = x.to(device).float()
    (
        embedding_loss,
        x_hat,
        perplexity,
        min_encoding,
        min_encoding_indices,
    ) = model(x)
    recon_loss = torch.mean((x_hat - x) ** 2)
    loss = recon_loss + embedding_loss
    pbar.set_description(f"Loss: {loss.item():.4f}")

    original_image = x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    gen_image = x_hat.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    concat_image = get_concat_h(
        Image.fromarray((original_image * 255).astype(np.uint8)),
        Image.fromarray((gen_image * 255).astype(np.uint8))
    )
    concat_image.save(f"results/test_{i}.png")

    if i == 10:
        break