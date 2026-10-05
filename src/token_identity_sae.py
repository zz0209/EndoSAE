import torch
from torch import nn
from torch.nn import functional as F


METHODS = ("token_sparse", "mean_sparse", "token_dense", "mean_dense", "raw_supcon")


class TokenIdentitySAE(nn.Module):
    def __init__(self, config, method):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.identity_space = config.get("identity_space", "projected")
        if self.identity_space not in ("projected", "code"):
            raise ValueError(self.identity_space)
        self.top_k = int(config["top_k"])
        self.latent_dim = int(config["latent_dim"])
        dimension = int(config["input_dim"])
        if method == "raw_supcon":
            self.projection = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(),
                                            nn.Linear(256, config["readout_dim"]))
        else:
            self.encoder = nn.Linear(dimension, self.latent_dim)
            self.decoder = nn.Linear(self.latent_dim, dimension)
            readout = nn.Linear(self.latent_dim, config["readout_dim"], bias=False)
            self.readout = readout if self.identity_space == "projected" else nn.Identity()
            with torch.no_grad():
                nn.init.normal_(self.decoder.weight, std=dimension ** -.5)
                self.normalize_decoder()
                self.encoder.weight.copy_(self.decoder.weight.T)
                self.encoder.bias.zero_()
                self.decoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        if self.method == "raw_supcon":
            return
        norms = self.decoder.weight.norm(dim=0, keepdim=True)
        if torch.any(norms <= 0) or not torch.isfinite(norms).all():
            raise ValueError("Invalid decoder column")
        self.decoder.weight.div_(norms)

    def encode(self, values):
        code = F.relu(self.encoder(values))
        if self.method.endswith("sparse"):
            selected, indices = torch.topk(code, self.top_k, dim=-1, sorted=False)
            code = torch.zeros_like(code).scatter(-1, indices, selected)
        return code

    def forward(self, tokens):
        mean = tokens.mean(dim=-2)
        if self.method == "raw_supcon":
            return F.normalize(self.projection(mean), dim=-1), None, None
        local = self.encode(tokens)
        pooled = local.mean(dim=-2) if self.method.startswith("token_") else self.encode(mean)
        return F.normalize(self.readout(pooled), dim=-1), self.decoder(local), local
