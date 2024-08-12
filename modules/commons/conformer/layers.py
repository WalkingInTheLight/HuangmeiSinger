from torch import nn
import torch

from modules.commons.layers import LayerNorm


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
    """

    def __init__(self, channels,out_channels,kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)

class multikernelConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
    """

    def __init__(self, channels, kernel_size1, kernel_size2,kernel_size3,kernel_size4,activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(multikernelConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size1 - 1) % 2 == 0
        assert (kernel_size2 - 1) % 2 == 0
        assert (kernel_size3 - 1) % 2 == 0
        assert (kernel_size4 - 1) % 2 == 0
        self.convmodule1 = ConvolutionModule(channels,channels//4,kernel_size1,activation,bias)
        self.convmodule2 = ConvolutionModule(channels,channels//4,kernel_size2,activation,bias)
        self.convmodule3 = ConvolutionModule(channels,channels//4,kernel_size3,activation,bias)
        self.convmodule4 = ConvolutionModule(channels,channels//4,kernel_size4,activation,bias)
        #self.convmodule5 = ConvolutionModule(channels, channels // 8, kernel_size5, activation, bias)
        #self.convmodule6 = ConvolutionModule(channels, channels // 8, kernel_size6, activation, bias)
        #self.convmodule7 = ConvolutionModule(channels, channels // 8, kernel_size7, activation, bias)
        #self.convmodule8 = ConvolutionModule(channels, channels // 8, kernel_size8, activation, bias)

    def forward(self, x):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # # exchange the temporal dimension and the feature dimension
        # x = x.transpose(1, 2)
        #
        # # GLU mechanism
        # x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        # x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)
        #
        # # 1D Depthwise Conv
        # x = self.depthwise_conv(x)
        # x = self.activation(self.norm(x))
        #
        # x = self.pointwise_conv2(x)
        #
        # return x.transpose(1, 2)
        x1 = self.convmodule1(x)
        x2 = self.convmodule2(x)
        x3 = self.convmodule3(x)
        x4 = self.convmodule4(x)
        #x5 = self.convmodule5(x)
        #x6 = self.convmodule6(x)
        #x7 = self.convmodule7(x)
        #x8 = self.convmodule8(x)
        x = torch.cat((x1,x2,x3,x4),dim=2)
        return x


# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))





class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.
    This is a module of multi-leyered conv1d designed
    to replace positionwise feed-forward network
    in Transforner block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize MultiLayeredConv1d module.
        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).
        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).
        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


class EncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
            self,
            size,
            self_attn,
            feed_forward,
            feed_forward_macaron,
            conv_module,
            dropout_rate,
            normalize_before=True,
            concat_after=False,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.
        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask



if __name__=='__main__':
    from modules.commons.conformer.espnet_transformer_attn import RelPositionMultiHeadedAttention
    from modules.commons.conformer.espnet_positional_embedding import PositionalEncoding
    hidden_size = 6
    batch_size = 2
    dropout_rate = 0.1
    time_step = 5
    sizeC = 6
    n_feat = d_model = 6
    n_head = 3
    dim_k = n_feat // n_head
    h = n_head
    ffm = None
    kernel_size = 5


    q = torch.randn(batch_size, time_step, sizeC)
    k = torch.randn(batch_size, time_step, sizeC)
    v = torch.randn(batch_size, time_step, sizeC)


    self_attn = RelPositionMultiHeadedAttention(n_head, hidden_size,dropout_rate)
    ffn = PositionalEncoding(d_model,dropout_rate)
    conv = ConvolutionModule(d_model,6,kernel_size)

    self_attn1 = RelPositionMultiHeadedAttention
    ffn1 = PositionalEncoding
    conv1 = ConvolutionModule

    import torch.nn.functional as F

    input_s = torch.Tensor([2,4]).to(torch.int32)
    valid_encoder_pos = torch.unsqueeze(
        torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, time_step - L)), 0) for L in input_s]), 2)
    print(valid_encoder_pos)
    valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
    print(valid_encoder_pos_matrix)
    invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
    mask = invalid_encoder_pos_matrix.to(torch.bool)
    print(mask.shape)
    #mask = torch.tensor([[True,True,True,True,False,False],[True,True,True,False,False,False]])
    mask = torch.randn(batch_size,time_step,d_model)
    print(mask)


    x = torch.randn(batch_size,time_step,d_model)
    y = torch.randn(1,time_step,d_model)
    z = torch.cat((x,y),dim=0)
    print(z)
    print(z.shape)
    #(x.shape[0], x.shape[1] - 1, size)
    print(z.shape[0])
    print(z.shape[1]-1)
    cache = torch.randn(batch_size,time_step - 1, d_model)
    print(cache.shape)
    pos = torch.randn(batch_size, time_step, sizeC)
    encoder = EncoderLayer(hidden_size, self_attn(q, k, v,pos, mask), ffn(x), ffn(x), conv(x), dropout_rate,)
    #encoder = EncoderLayer(hidden_size,self_attn1,ffn1,ffn1,conv1,dropout_rate)

    e = encoder(x, mask, cache)
    print("e_shape:",e.shape)
    print("e:",e)