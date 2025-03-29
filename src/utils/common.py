import numpy as np
import torch

from tqdm import tqdm


def load_model(model: torch.nn.Module, state_dict_path: str) -> torch.nn.Module:
    """
    Load the model with its respective weight file

    Args:
        model (torch.nn.Module): Model to load
        state_dict_path(str): Path to the state dict file
    
    Returns:
        torch.nn.Module: Model with loaded weights

    """

    model.load_state_dict(torch.load(state_dict_path))
    return model

def extract_tokens_from_vqvae(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> list:
    """
    Extract tokens from the VQVAE model
    
    Args: 
        model (torch.nn.Module):  VQVAE model loaded with weights
        dataloader (torch.utils.data.DataLoader): Dataloader with the dataset
    
    Saves the tokens to a .npy file separately for each batch
    """

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    json_list = []

    for i, (x, y) in pbar:
        x = x.to(device).float()
        (
            embedding_loss,
            x_hat,
            perplexity,
            min_encoding,
            min_encoding_indices,
        ) = model(x)

        json_list.append(
            {
                "image_id": i,
                "image": x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),
                "tokens": min_encoding_indices.squeeze(0).cpu().detach().numpy(),
                "perplexity": perplexity.item(),
            }
        )

    return json_list