import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_h_dim, hid_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, res_h_dim, num_layers):
        super(ResidualBlock, self).__init__()
        self.num_layers = num_layers
        self.res_stack = nn.ModuleList(
            [
                ResidualLayer(in_dim=in_dim, hid_dim=hid_dim, res_h_dim=res_h_dim)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.res_stack:
            x = layer(x)

        x = F.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, res_h_dim, num_layers):
        super(Encoder, self).__init__()

        kernel = 4
        stride = 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_dim, hid_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                hid_dim // 2, hid_dim, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                hid_dim, hid_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            ResidualBlock(
                in_dim=hid_dim,
                hid_dim=hid_dim,
                res_h_dim=res_h_dim,
                num_layers=num_layers,
            ),
        )

    def forward(self, x):
        return self.conv_stack(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # print(z_flattened.shape)
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # print(min_encoding_indices.shape)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings
        ).to(device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # print(min_encodings.shape)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, res_h_dim, num_layers):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, hid_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            ResidualBlock(
                in_dim=hid_dim,
                hid_dim=hid_dim,
                res_h_dim=res_h_dim,
                num_layers=num_layers,
            ),
            nn.ConvTranspose2d(
                hid_dim, hid_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hid_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1
            ),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta,
        save_img_embedding_map=False,
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        (
            embedding_loss,
            z_q,
            perplexity,
            min_codings,
            coding_indices,
        ) = self.vector_quantization(z_e)
        # print(coding_indices.shape)
        x_hat = self.decoder(z_q)

        if verbose:
            print("original data shape:", x.shape)
            print("encoded data shape:", z_e.shape)
            print("recon data shape:", x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, min_codings, coding_indices
