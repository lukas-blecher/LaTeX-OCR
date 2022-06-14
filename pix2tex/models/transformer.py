import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p
from x_transformers import TransformerWrapper, Decoder

from .embed import AbsolutePositionalEmbedding


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len=256, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            # print('arw:',out.shape)
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out


class LatexTransformerDecoder(nn.Module):
    """
        Transformer Decoder implemented by PyTorch's Transformer
        NOTE: 
            The deoder's output is the logits, not the loss.
            If using this decoder, loss function should be followed,
            I removed loss from decoder because this makes it more 
            convenient to convert the model to ONNX format.
    """
    def __init__(self, model_dim: int=512, nhead: int=8, ff_dim: int=2048,
                 dropout: float=0.0, depth: int=4, num_tokens: int=8000,
                 max_seq_len: int=512, norm_eps: float=1e-6, activation=F.gelu,
                 ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, model_dim, padding_idx=0)
        self.pos_emb = AbsolutePositionalEmbedding(max_seq_len, model_dim)
        norm_layer = nn.LayerNorm(model_dim, eps=norm_eps)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=model_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=depth,
            norm=norm_layer,
        )

        self.to_logits = nn.Linear(model_dim, num_tokens)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
            memory here is the output of encoder, shape:
            tgt: (B, T)
            tgt_mask: (T, T)
        """
        tgt = self.token_emb(tgt)
        tgt += self.pos_emb(tgt)
        out = self.transformer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        out = self.to_logits(out)  # B * num_tokens
        return out


def get_decoder(args):
    return CustomARWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                **args.decoder_args
            )),
        pad_value=args.pad_token)
