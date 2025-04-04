import torch
import torch.nn as nn
from tqdm import tqdm
import torch.serialization


class CausalTransformer(nn.Module):
    def __init__(self, d_model=300, nhead=6, num_layers=6, dim_feedforward=512, max_len=128):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, pos_ids):
        x = x + self.pos_embedding(pos_ids)

        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len),
                          diagonal=1).bool().to(x.device)

        x = x.transpose(0, 1)  # (seq_len, batch, d_model)

        memory = self.encoder(x, mask=mask)
        output = self.decoder(x, memory, tgt_mask=mask, memory_mask=mask)
        output = output.transpose(0, 1)  # (batch, seq_len, d_model)

        return self.fc_out(output)


# Add custom class to safe globals
# torch.serialization.add_safe_globals([CausalTransformer])

# Load model
model = torch.load('causal_transformer_full.pth', weights_only=False)


# Save state dict
torch.save(model.state_dict(), 'causal_transformer_state_dict.pth')
print("Model state dictionary saved successfully")
