"""Three-curve convolution/attention architecture with separate typed weights.

Local convolutions retain peak detail before attention links distant regions.
Do not load this architecture through the legacy eight-channel weight reader.
Candidate pruning requires separate scientific qualification.
"""


def network(class_count, *, attention=True, tokens=128, architecture=None):
    import torch
    from torch import nn
    from .structural_attention_pool import adaptive
    from .structural_attention_weights import ARCHITECTURE,ARCHITECTURES
    architecture=ARCHITECTURE if architecture is None else architecture
    if (type(class_count) is not int or not 2 <= class_count <= 230
            or type(attention) is not bool or tokens not in (64, 128, 256)
            or not isinstance(architecture,str) or architecture not in ARCHITECTURES):
        raise ValueError('Explicit bounded three-curve attention architecture required')

    class AveragePool(nn.Module):
        def forward(self,values):return adaptive(values,tokens)

    class ThreeCurveModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.local = nn.Sequential(nn.Conv1d(3, 24, 9, padding=4), nn.GELU(),
                nn.Conv1d(24, 32, 7, stride=2, padding=3), nn.GELU(),
                nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU())
            self.pool = AveragePool()
            self.use_attention = attention
            if attention:
                # All examples share declared physical Q edges. Positions
                # are encoded inside the model, not a fourth input channel.
                self.position = nn.Parameter(torch.zeros(1, tokens, 32))
                layer = nn.TransformerEncoderLayer(d_model=32, nhead=4,
                    dim_feedforward=96, dropout=.1, activation='gelu',
                    batch_first=True, norm_first=False)
                self.attention = nn.TransformerEncoder(layer, num_layers=1,
                    enable_nested_tensor=False)
            layers=[nn.Flatten(),nn.Linear(32*tokens,64)]
            if architecture==ARCHITECTURE:
                # The unnormalized hidden layer can become entirely negative
                # on a many-family curriculum, saturating GELU to exact zero.
                # Normalize before that nonlinearity, preserving all three
                # physical input channels and the attention encoder.
                layers.append(nn.LayerNorm(64))
            self.head=nn.Sequential(*layers,nn.GELU(),nn.Dropout(.1),nn.Linear(64,class_count))

        def forward(self, values, coverage=None):
            if values.ndim != 3 or values.shape[1] != 3:
                raise ValueError('Exactly three ordered curve channels required')
            mask = None
            if coverage is not None:
                if coverage.shape != (values.shape[0], values.shape[2]):
                    raise ValueError('Coverage metadata must match the physical Q grid')
                if not torch.isfinite(coverage).all() or ((coverage < 0) | (coverage > 1)).any():
                    raise ValueError('Finite physical coverage between zero and one required')
                if not (coverage > 0).any(dim=1).all():
                    raise ValueError('An entirely masked pattern cannot be ranked')
                values = values.masked_fill((coverage == 0).unsqueeze(1), 0.)
                mask = adaptive(coverage.unsqueeze(1),tokens,maximum=True).squeeze(1) == 0
            encoded = self.pool(self.local(values))
            if self.use_attention:
                encoded = self.attention(encoded.transpose(1, 2)+self.position,
                    src_key_padding_mask=mask).transpose(1, 2)
            if mask is not None:
                encoded = encoded.masked_fill(mask.unsqueeze(1), 0.)
            return self.head(encoded)

    return ThreeCurveModel()
