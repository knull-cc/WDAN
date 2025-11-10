import torch
import torch.nn as nn
import torch.nn.functional as F
from nns.tsl_layers.Transformer_Encoder import Encoder, EncoderLayer
from nns.tsl_layers.SWTAttention_Family import GeomAttentionLayer, GeomAttention
from nns.tsl_layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self,seq_len,
                 pred_len,
                 output_attention=False,
                 use_norm=1,
                 geomattn_dropout=0.1,
                 alpha=0.3,
                 kernel_size=3,
                 d_model=512,
                 embed='timeF',
                 freq='h',
                 dropout=0.1,
                 factor=1,
                 requires_grad=True,
                 wv='db1',
                 m=3,
                 dec_in=7,
                 e_layers=2,
                 d_ff=2048,
                 activation='gelu'):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.geomattn_dropout = geomattn_dropout
        self.alpha = alpha
        self.kernel_size = kernel_size

        enc_embedding = DataEmbedding_inverted(seq_len, d_model, 
                                               embed, freq, dropout)
        self.enc_embedding = enc_embedding

        encoder = Encoder(
            [  
                EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False, factor, attention_dropout=dropout, 
                            output_attention=output_attention, alpha=self.alpha
                        ),
                        d_model, 
                        requires_grad=requires_grad, 
                        wv=wv, 
                        m=m, 
                        d_channel=dec_in, 
                        kernel_size=self.kernel_size, 
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(e_layers) 
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder = encoder

        projector = nn.Linear(d_model, self.pred_len, bias=True)
        self.projector = projector

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # x_enc /= stdev
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        enc_embedding = self.enc_embedding
        encoder = self.encoder
        projector = self.projector
        # Linear Projection             B L N -> B L' (pseudo temporal tokens) N 
        enc_out = enc_embedding(x_enc, x_mark_enc) 

        # SimpleTM Layer                B L' N -> B L' N 
        enc_out, attns = encoder(enc_out, attn_mask=None)

        # Output Projection             B L' N -> B H (Horizon) N
        dec_out = projector(enc_out).permute(0, 2, 1)[:, :, :N] 

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, None, None, None)
        return dec_out, attns 