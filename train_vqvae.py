import mlflow.pytorch
import numpy as np
import torch

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
batch_size = 350
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
trainloader, testloader = get_cifar10_loader(batch_size)

# Load model
model = VQVAE(
    n_hidden, n_residual_hidden, n_residual_layers, n_embeddings, embedding_dim, beta
).to(device)


from tqdm import tqdm
import mlflow.pytorch

results = {
    "recon_errors": [],
    "perplexities": [],
    "embedding_errors": [],
    "loss_vals": [],
    "n_updates": 0,
}


def reset_results():
    results["recon_errors"] = []
    results["embedding_errors"] = []
    results["perplexities"] = []
    results["loss_vals"] = []


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


def train(trainloader, testloader):
    best_loss = float("inf")
    best_model_dict = {}

    with mlflow.start_run():

        for i in range(num_epochs):
            model.train()
            pbar = tqdm(enumerate(trainloader), total=len(trainloader))
            for j, (x, _) in pbar:

                x = x.to(device).float()

                optimizer.zero_grad()

                (
                    embedding_loss,
                    x_hat,
                    perplexity,
                    min_encoding,
                    min_encoding_indices,
                ) = model(x)
                recon_loss = torch.mean((x_hat - x) ** 2)
                loss = recon_loss + embedding_loss

                loss.backward()
                optimizer.step()

                results["recon_errors"].append(recon_loss.cpu().detach().numpy())
                results["embedding_errors"].append(
                    embedding_loss.cpu().detach().numpy()
                )
                results["perplexities"].append(perplexity.cpu().detach().numpy())
                results["loss_vals"].append(loss.cpu().detach().numpy())
                results["n_updates"] = i * len(trainloader) + j

            epoch_recon_error = np.mean(results["recon_errors"])
            epoch_embedding_error = np.mean(results["embedding_errors"])
            epoch_loss = np.mean(results["loss_vals"])
            epoch_perplexity = np.mean(results["perplexities"])

            mlflow.log_metric("recon_error", epoch_recon_error, step=i)
            mlflow.log_metric("embedding_error", epoch_embedding_error, step=i)
            mlflow.log_metric("loss", epoch_loss, step=i)
            mlflow.log_metric("perplexity", epoch_perplexity, step=i)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    model.state_dict(),
                    f"best_models/v5/vqvae_model_aircraft_v5_loss_{best_loss:.4f}.pth",
                )

            reset_results()

            # print per epoch
            print(
                f"Epoch {i} - Recon Error: {epoch_recon_error}, Embedding Error: {epoch_embedding_error}, Loss: {epoch_loss}, Perplexity: {epoch_perplexity}"
            )

            print(f"Testing - Epoch: {i}")
            model.eval()

            with torch.no_grad():
                test_recon_errors = []
                test_perplexities = []
                test_embedding_errors = []
                test_losses = []
                for x, _ in testloader:
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

                    test_recon_errors.append(recon_loss.cpu().detach().numpy())
                    test_embedding_errors.append(embedding_loss.cpu().detach().numpy())
                    test_perplexities.append(perplexity.cpu().detach().numpy())
                    test_losses.append(loss.cpu().detach().numpy())

                test_recon_error = np.mean(test_recon_errors)
                test_embedding_error = np.mean(test_embedding_errors)
                test_loss = np.mean(test_losses)
                test_perplexity = np.mean(test_perplexities)

                mlflow.log_metric("test_recon_error", test_recon_error, step=i)
                mlflow.log_metric("test_embedding_error", test_embedding_error, step=i)
                mlflow.log_metric("test_loss", test_loss, step=i)
                mlflow.log_metric("test_perplexity", test_perplexity, step=i)

                print(
                    f"Test Epoch {i} - Recon Error: {test_recon_error}, Embedding Error: {test_embedding_error}, Loss: {test_loss}, Perplexity: {test_perplexity}"
                )

            lr_scheduler.step()

if __name__ == "__main__":
    train(trainloader, testloader)