import torch.nn as nn
import torch
import config


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(in_features, 2 * in_features)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(2 * in_features, out_features)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.out_proj(self.relu(self.in_proj(x)))


class Transformer(nn.Module):
    def __init__(self, model_args: config.ModelArgs):
        super().__init__()

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # Using a ModuleDict lets us delete layers witout affecting names,
        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = nn.TransformerDecoderLayer(
                model_args.dim, model_args.n_heads
            )

        self.norm = nn.LayerNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size)

    def forward(self, tokens: torch.Tensor):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens.float()

        for layer in self.layers.values():
            h = layer(h, h)

        h = self.norm(h) if self.norm else h
        output = self.output(h).clone() if self.output else h
        return output
