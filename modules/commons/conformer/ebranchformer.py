# Copyright 2022 Kwangyoun Kim (ASAPP inc.)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""E-Branchformer encoder definition.
Reference:
    Kwangyoun Kim, Felix Wu, Yifan Peng, Jing Pan,
    Prashant Sridhar, Kyu J. Han, Shinji Watanabe,
    "E-Branchformer: Branchformer with Enhanced merging
    for speech recognition," in SLT 2022.
"""
import logging
from typing import List, Optional, Tuple, Union

import numpy
import torch
from torch import nn

from modules.commons.common_layers import FastSelfAttention, ConvolutionalGatingMLP
from modules.commons.conformer.espnet_positional_embedding import RelPositionalEncoding
from modules.commons.conformer.espnet_transformer_attn import RelPositionMultiHeadedAttention
from modules.commons.conformer.layers import PositionwiseFeedForward,Swish


class EBranchformerEncoderLayer(torch.nn.Module):
    """E-Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention
        cgmlp: ConvolutionalGatingMLP
        feed_forward: feed-forward module, optional
        feed_forward: macaron-style feed-forward module, optional
        dropout_rate (float): dropout probability
        merge_conv_kernel (int): kernel size of the depth-wise conv in merge module
    """

    def __init__(
        self,
        size: int,
        attn: torch.nn.Module,
        cgmlp: torch.nn.Module,
        feed_forward: Optional[torch.nn.Module],
        feed_forward_macaron: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_conv_kernel: int = 3,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = nn.LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = nn.LayerNorm(size)

        self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        self.norm_mlp = nn.LayerNorm(size)  # for the MLP module
        self.norm_final = nn.LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            size + size,
            size + size,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=size + size,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.norm_mha(x1)

        if isinstance(self.attn, FastSelfAttention):
            x_att = self.attn(x1, mask)
        else:
            if pos_emb is not None:
                x_att = self.attn(x1, x1, x1, pos_emb, mask)
            else:
                x_att = self.attn(x1, x1, x1, mask)

        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        x2 = self.norm_mlp(x2)

        if pos_emb is not None:
            x2 = (x2, pos_emb)
        x2 = self.cgmlp(x2, mask)
        if isinstance(x2, tuple):
            x2 = x2[0]

        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + self.dropout(self.merge_proj(x_concat + x_tmp))

        if self.feed_forward is not None:
            # feed forward module
            residual = x
            x = self.norm_ff(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class EBranchformerEncoder_A(torch.nn.Module):
    """E-Branchformer encoder module."""
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 12,
        kernel_size: int = 9,
        num_heads: int = 4,
        cgmlp_linear_units: int = 256 * 4,  # 2048,
        cgmlp_conv_kernel: int = 31,  # 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "Swich",  # "identity",
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        max_pos_emb_len: int = 5000,
        linear_units: int = 256 * 4,  # 2048,
        merge_conv_kernel: int = 3,
    ):
        super().__init__()
        self._output_size = output_size = hidden_size
        attention_heads = num_heads
        num_blocks = num_layers
        # linear_units = hidden_size * 4


        pos_enc_class = RelPositionalEncoding
        # self.pos_embed = pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)

        self.pos_embed = torch.nn.Sequential(
            pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
        )

        activation = Swish()
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )

        use_rel = False # None
        if use_rel:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                dropout_rate,
            )
        else:
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                output_size,
                attention_heads,
                dropout_rate,
            )


        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        self.encoder_layers = nn.ModuleList([EBranchformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                cgmlp_layer(*cgmlp_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                merge_conv_kernel,
        ) for _ in range(num_blocks)])
        self.after_norm = nn.LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs_pad):
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): Intermediate CTC module.
            max_layer (int): Layer depth below which InterCTC is applied.
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """

        nonpadding_mask = xs_pad.abs().sum(-1) > 0

        xs_pad = self.pos_embed(xs_pad)

        for l in self.encoder_layers:
            xs_pad, masks = l(xs_pad, nonpadding_mask[:, None, :])


        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad) * nonpadding_mask.float()[:, :, None]
        return xs_pad


class EBranchformerEncoder(EBranchformerEncoder_A):
    def __init__(self, embed_tokens, hidden_size=256, num_layers=4, kernel_size=9, num_heads=4):
        # super().__init__(hidden_size, num_layers, k1,k2,k3,k4, num_heads=num_heads)
        super().__init__(hidden_size=hidden_size, num_layers=num_layers, kernel_size=kernel_size, num_heads=num_heads)
    #
    # def __init__(self, embed_tokens, hidden_size):
    #     super().__init__(hidden_size)
    #     #super().__init__(hidden_size, num_layers,num_heads=num_heads)
        self.embed = embed_tokens   # self.embed = Embedding(dict_size, hidden_size, padding_idx=0)

    def forward(self, x):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = self.embed(x)  # [B, T, H]
        x = super(EBranchformerEncoder_A, self).forward(x)
        return x

class EBranchformerDecoder(EBranchformerEncoder_A):
    def __init__(self, hidden_size=256, num_layers=4, kernel_size=9, num_heads=4):
        # super().__init__(hidden_size, num_layers, k1,k2,k3,k4, num_heads=num_heads)
        super().__init__(hidden_size=hidden_size, num_layers=num_layers, kernel_size=kernel_size, num_heads=num_heads)