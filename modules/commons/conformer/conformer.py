from torch import nn
from modules.commons.conformer.espnet_positional_embedding import RelPositionalEncoding
from modules.commons.conformer.espnet_transformer_attn import RelPositionMultiHeadedAttention
from modules.commons.conformer.layers import Swish, ConvolutionModule, EncoderLayer, MultiLayeredConv1d, multikernelConvolutionModule
from modules.commons.layers import Embedding


class ConformerLayers(nn.Module):
    # def __init__(self, hidden_size, num_layers, k1=9,k2=9,k3=9,k4=9, dropout=0.1, num_heads=4,
    def __init__(self, hidden_size, num_layers, kernel_size=31, dropout=0.1, num_heads=4,
                 use_last_norm=True, save_hidden=False):
        super().__init__()
        self.use_last_norm = use_last_norm
        self.layers = nn.ModuleList()
        activation = Swish()
        linear_units = hidden_size * 4  # 2048

        kernel_size = 31
        # ```
        # idim,
        # attention_dim = 256,
        # attention_heads = 4,
        # linear_units = 2048,
        # num_blocks = 6,
        # dropout_rate = 0.1,
        # positional_dropout_rate = 0.1,
        # attention_dropout_rate = 0.0,
        # input_layer = "conv2d",
        # normalize_before = True,
        # concat_after = False,
        # positionwise_layer_type = "linear",
        # positionwise_conv_kernel_size = 1,
        # cnn_module_kernel = 31,
        # ```
        # self-attention module definition
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (num_heads, hidden_size, dropout)

        # feed-forward module definition
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (hidden_size, linear_units, 1, dropout)

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (hidden_size, hidden_size, kernel_size, activation)

        # 这个可以改
        self.pos_embed = RelPositionalEncoding(hidden_size, dropout)
        # 如
        # self.embed = torch.nn.Sequential(
        #     torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
        #     pos_enc_class(attention_dim, positional_dropout_rate),
        # )
        self.encoder_layers = nn.ModuleList([EncoderLayer(
            hidden_size,
            encoder_selfattn_layer(*encoder_selfattn_layer_args),
            positionwise_layer(*positionwise_layer_args),
            positionwise_layer(*positionwise_layer_args),   # if macaron_style else None,
            convolution_layer(*convolution_layer_args),  #  if use_cnn_module else None,
            # multikernelConvolutionModule(hidden_size,k1,k2,k3,k4,Swish()),
            dropout,
        ) for _ in range(num_layers)])
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Linear(hidden_size, hidden_size)
        self.save_hidden = save_hidden
        if save_hidden:
            self.hiddens = []

    def forward(self, x, padding_mask=None):
        """

        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: [B, T, H]
        """
        self.hiddens = []
        nonpadding_mask = x.abs().sum(-1) > 0
        x = self.pos_embed(x)
        # print(x[0].shape, x[1].shape)
        for l in self.encoder_layers:
            x, mask = l(x, nonpadding_mask[:, None, :])
            if self.save_hidden:
                self.hiddens.append(x[0])

        if isinstance(x, tuple):
            x = x[0]
        x = self.layer_norm(x) * nonpadding_mask.float()[:, :, None]
        return x


class ConformerEncoder(ConformerLayers):
    def __init__(self, embed_tokens, hidden_size=256, num_layers=4, kernel_size=9, num_heads=4):  # kernel_size=31
        # super().__init__(hidden_size, num_layers, k1,k2,k3,k4, num_heads=num_heads)
        super().__init__(hidden_size=hidden_size, num_layers=num_layers, kernel_size=kernel_size, num_heads=num_heads)
        self.embed = embed_tokens   # self.embed = Embedding(dict_size, hidden_size, padding_idx=0)

    def forward(self, x):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = self.embed(x)  # [B, T, H]
        x = super(ConformerEncoder, self).forward(x)
        return x


class ConformerDecoder(ConformerLayers):
    def __init__(self, hidden_size=256, num_layers=4, kernel_size=9, num_heads=4):  # kernel_size=31
        super().__init__(hidden_size=hidden_size, num_layers=num_layers, kernel_size=kernel_size, num_heads=num_heads)

import torch
if __name__ == '__main__':
    con = ConformerLayers(8,9)
    y = torch.randn(2, 6, 8)
    mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, False, False, False]])
    c = con(y, mask)
    print(c)
    print(c.shape)

    ce = ConformerEncoder(8,9,3)
    y = torch.randint(2, (6,8))
    cee = ce(y)
    print("cee",cee)

    nonpadding_mask = y.abs().sum(-1) > 0
    print(nonpadding_mask)






