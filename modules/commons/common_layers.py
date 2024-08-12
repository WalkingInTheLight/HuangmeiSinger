import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.onnx.operators
import torch.nn.functional as F
import utils


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(self.args)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(input, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.last_attn_probs = None

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query, key, value,
            key_padding_mask=None,
            incremental_state=None,
            need_weights=True,
            static_kv=False,
            attn_mask=None,
            before_softmax=False,
            need_head_weights=False,
            enc_dec_attn_constraint_mask=None,
            reset_attn_weight=None
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if self.enable_torch_version and incremental_state is None and not static_kv and reset_attn_weight is None:
            if self.qkv_same_dim:
                return F.multi_head_attention_forward(query, key, value,
                                                      self.embed_dim, self.num_heads,
                                                      self.in_proj_weight,
                                                      self.in_proj_bias, self.bias_k, self.bias_v,
                                                      self.add_zero_attn, self.dropout,
                                                      self.out_proj.weight, self.out_proj.bias,
                                                      self.training, key_padding_mask, need_weights,
                                                      attn_mask)
            else:
                return F.multi_head_attention_forward(query, key, value,
                                                      self.embed_dim, self.num_heads,
                                                      torch.empty([0]),
                                                      self.in_proj_bias, self.bias_k, self.bias_v,
                                                      self.add_zero_attn, self.dropout,
                                                      self.out_proj.weight, self.out_proj.bias,
                                                      self.training, key_padding_mask, need_weights,
                                                      attn_mask, use_separate_proj_weight=True,
                                                      q_proj_weight=self.q_proj_weight,
                                                      k_proj_weight=self.k_proj_weight,
                                                      v_proj_weight=self.v_proj_weight)

        if incremental_state is not None:
            print('Not implemented error.')
            exit()
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            print('Not implemented error.')
            exit()

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif len(attn_mask.shape) == 3:
                attn_mask = attn_mask[:, None].repeat([1, self.num_heads, 1, 1]).reshape(
                    bsz * self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attn_mask

        if enc_dec_attn_constraint_mask is not None:  # bs x head x L_kv
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                enc_dec_attn_constraint_mask.unsqueeze(2).bool(),
                -1e9,
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -1e9,
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

        if reset_attn_weight is not None:
            if reset_attn_weight:
                self.last_attn_probs = attn_probs.detach()
            else:
                assert self.last_attn_probs is not None
                attn_probs = self.last_attn_probs
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, (attn_weights, attn_logits)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)


class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, padding="SAME", kernel_size=1, dropout=0., act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(hidden_size, filter_size, kernel_size)
            )
        self.ffn_2 = Linear(filter_size, hidden_size)
        if self.act == 'swish':
            self.swish_fn = CustomSwish()

    def forward(self, x, incremental_state=None):
        # x: T x B x C
        if incremental_state is not None:
            assert incremental_state is None, 'Nar-generation does not allow this.'
            exit(1)

        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5

        if incremental_state is not None:
            x = x[-1:]
        if self.act == 'gelu':
            x = F.gelu(x)
        if self.act == 'relu':
            x = F.relu(x)
        if self.act == 'swish':
            x = self.swish_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class BatchNorm1dTBC(nn.Module):
    def __init__(self, c):
        super(BatchNorm1dTBC, self).__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        """

        :param x: [T, B, C]
        :return: [T, B, C]
        """
        x = x.permute(1, 2, 0)  # [B, C, T]
        x = self.bn(x)  # [B, C, T]
        x = x.permute(2, 0, 1)  # [T, B, C]
        return x


class EncSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, padding='SAME', norm='ln', act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            if norm == 'ln':
                self.layer_norm1 = LayerNorm(c)
            elif norm == 'bn':
                self.layer_norm1 = BatchNorm1dTBC(c)
            self.self_attn = MultiheadAttention(
                self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False,
            )
        if norm == 'ln':
            self.layer_norm2 = LayerNorm(c)
        elif norm == 'bn':
            self.layer_norm2 = BatchNorm1dTBC(c)
        self.ffn = TransformerFFNLayer(
            c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding, act=act)

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x)
            x, _, = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class DecSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)
        self.self_attn = MultiheadAttention(
            c, num_heads, self_attention=True, dropout=attention_dropout, bias=False
        )
        self.layer_norm2 = LayerNorm(c)
        self.encoder_attn = MultiheadAttention(
            c, num_heads, encoder_decoder_attention=True, dropout=attention_dropout, bias=False,
        )
        self.layer_norm3 = LayerNorm(c)
        self.ffn = TransformerFFNLayer(
            c, 4 * c, padding='LEFT', kernel_size=kernel_size, dropout=relu_dropout, act=act)

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            attn_out=None,
            reset_attn_weight=None,
            **kwargs,
    ):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
            self.layer_norm3.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        if encoder_out is not None:
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                enc_dec_attn_constraint_mask=None, #utils.get_incremental_state(self, incremental_state, 'enc_dec_attn_constraint_mask'),
                reset_attn_weight=reset_attn_weight
            )
            attn_logits = attn[1]
        else:
            assert attn_out is not None
            x = self.encoder_attn.in_proj_v(attn_out.transpose(0, 1))
            attn_logits = None
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm3(x)
        x = self.ffn(x, incremental_state=incremental_state)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        # if len(attn_logits.size()) > 3:
        #    indices = attn_logits.softmax(-1).max(-1).values.sum(-1).argmax(-1)
        #    attn_logits = attn_logits.gather(1,
        #        indices[:, None, None, None].repeat(1, 1, attn_logits.size(-2), attn_logits.size(-1))).squeeze(1)
        return x, attn_logits



"""Fastformer attention definition.

Reference:
    Wu et al., "Fastformer: Additive Attention Can Be All You Need"
    https://arxiv.org/abs/2108.09084
    https://github.com/wuch15/Fastformer

"""

import numpy
import torch


class FastSelfAttention(torch.nn.Module):
    """Fast self-attention used in Fastformer."""

    def __init__(
        self,
        size,
        attention_heads,
        dropout_rate,
    ):
        super().__init__()
        if size % attention_heads != 0:
            raise ValueError(
                f"Hidden size ({size}) is not an integer multiple "
                f"of attention heads ({attention_heads})"
            )
        self.attention_head_size = size // attention_heads
        self.num_attention_heads = attention_heads

        self.query = torch.nn.Linear(size, size)
        self.query_att = torch.nn.Linear(size, attention_heads)
        self.key = torch.nn.Linear(size, size)
        self.key_att = torch.nn.Linear(size, attention_heads)
        self.transform = torch.nn.Linear(size, size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        """Reshape and transpose to compute scores.

        Args:
            x: (batch, time, size = n_heads * attn_dim)

        Returns:
            (batch, n_heads, time, attn_dim)
        """

        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.reshape(*new_x_shape).transpose(1, 2)

    def forward(self, xs_pad, mask):
        """Forward method.

        Args:
            xs_pad: (batch, time, size = n_heads * attn_dim)
            mask: (batch, 1, time), nonpadding is 1, padding is 0

        Returns:
            torch.Tensor: (batch, time, size)
        """

        batch_size, seq_len, _ = xs_pad.shape
        mixed_query_layer = self.query(xs_pad)  # (batch, time, size)
        mixed_key_layer = self.key(xs_pad)  # (batch, time, size)

        if mask is not None:
            mask = mask.eq(0)  # padding is 1, nonpadding is 0

        # (batch, n_heads, time)
        query_for_score = (
            self.query_att(mixed_query_layer).transpose(1, 2)
            / self.attention_head_size**0.5
        )
        if mask is not None:
            min_value = float(
                numpy.finfo(
                    torch.tensor(0, dtype=query_for_score.dtype).numpy().dtype
                ).min
            )
            query_for_score = query_for_score.masked_fill(mask, min_value)
            query_weight = torch.softmax(query_for_score, dim=-1).masked_fill(mask, 0.0)
        else:
            query_weight = torch.softmax(query_for_score, dim=-1)

        query_weight = query_weight.unsqueeze(2)  # (batch, n_heads, 1, time)
        query_layer = self.transpose_for_scores(
            mixed_query_layer
        )  # (batch, n_heads, time, attn_dim)

        pooled_query = (
            torch.matmul(query_weight, query_layer)
            .transpose(1, 2)
            .reshape(-1, 1, self.num_attention_heads * self.attention_head_size)
        )  # (batch, 1, size = n_heads * attn_dim)
        pooled_query = self.dropout(pooled_query)
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)  # (batch, time, size)

        mixed_query_key_layer = (
            mixed_key_layer * pooled_query_repeat
        )  # (batch, time, size)

        # (batch, n_heads, time)
        query_key_score = (
            self.key_att(mixed_query_key_layer) / self.attention_head_size**0.5
        ).transpose(1, 2)
        if mask is not None:
            min_value = float(
                numpy.finfo(
                    torch.tensor(0, dtype=query_key_score.dtype).numpy().dtype
                ).min
            )
            query_key_score = query_key_score.masked_fill(mask, min_value)
            query_key_weight = torch.softmax(query_key_score, dim=-1).masked_fill(
                mask, 0.0
            )
        else:
            query_key_weight = torch.softmax(query_key_score, dim=-1)

        query_key_weight = query_key_weight.unsqueeze(2)  # (batch, n_heads, 1, time)
        key_layer = self.transpose_for_scores(
            mixed_query_key_layer
        )  # (batch, n_heads, time, attn_dim)
        pooled_key = torch.matmul(
            query_key_weight, key_layer
        )  # (batch, n_heads, 1, attn_dim)
        pooled_key = self.dropout(pooled_key)

        # NOTE: value = query, due to param sharing
        weighted_value = (pooled_key * query_layer).transpose(
            1, 2
        )  # (batch, time, n_heads, attn_dim)
        weighted_value = weighted_value.reshape(
            weighted_value.shape[:-2]
            + (self.num_attention_heads * self.attention_head_size,)
        )  # (batch, time, size)
        weighted_value = (
            self.dropout(self.transform(weighted_value)) + mixed_query_layer
        )

        return weighted_value


"""MLP with convolutional gating (cgMLP) definition.

References:
    https://openreview.net/forum?id=RA-zVvZLYIy
    https://arxiv.org/abs/2105.08050

"""

import torch.nn
from modules.commons.conformer.layers import Swish

class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = nn.LayerNorm(n_channels)
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            (kernel_size - 1) // 2,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = Swish()   # torch.nn.ReLU

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-6)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """

        x_r, x_g = x.chunk(2, dim=-1)

        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, mask):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None

        xs_pad = self.channel_proj1(xs_pad)  # size -> linear_units
        xs_pad = self.csgu(xs_pad)  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)  # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out



##############  DCMHAttention
from torch import Tensor
from typing import Optional,Tuple,List
from collections import namedtuple

class DCMHAttention(nn.Module):
    def __init__(self, config, lidx, use_sw=False): # : DCFormerConfig
        super().__init__()
        assert config.dim % config.n_head == 0
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.lidx = lidx
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.is_training = config.is_training
        self.dim = config.dim
        self.use_dcmha = config.use_dcmha
        self.scale_factor = 1 / math.sqrt(self.head_dim)
        self.q_chunk_size = config.q_chunk_size
        self.use_sw = use_sw
        self.dyn_w_proj = DynamicWeightProjection(num_heads=self.n_head, query_input_dim=config.dim,
                                                  dynamic_squeeze_ratio=self.n_head // 2,
                                                  dynamic_w_hidden_dim=self.n_head * 4, use_sw=use_sw)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSnorm(hid_dim=self.head_dim)
            self.k_norm = RMSnorm(hid_dim=self.head_dim)

        self.window_types = {
            "LG": [256, None],
            "LGLL": [256, None, 256, 256],
            "LGL6": [256, None, 256, 256, 256, 256, 256, 256],
        }

        self.query_wise = config.query_wise
        if config.window_type is None:  # LG
            self.window_size = None if self.lidx % 2 == 1 else config.window_size
        else:
            window_l = self.window_types[config.window_type]
            self.window_size = window_l[self.lidx % len(window_l)]

        if not self.is_training:
            self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def _generate_fast(self, x, input_pos, q, k, v, k_mask):
        B, T, D = x.shape
        N, I = self.n_head, self.dyn_w_proj.dynamic_hidden_dim  # 32, 2
        dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2 * 2 * N * (2 * I), 2 * 2 * N * 1],
                                                         -1)  # BTD, D(4K+4N) -> BT(4K+4N) -> BT(4K), BT(4N)
        dw_hidden = dw_hidden.view((B, T, 4, -1, 1))  # BT(4K) -> BT4K1
        dw = (self.dyn_w_proj.dw_hidden_activation(dw_hidden) * self.dyn_w_proj.qkw_m).sum(
            -2)  # gelu, BT4K1, 4K(IM)->BT4K(IM)->BT4(IM)
        w1, w2 = dw.view((B, T, 2, 2, -1, N)).split(I, -2)  # BT4(IM)->BT{pre/post}{q/k}IM->[BT22IM] * 2
        w1 = self.dyn_w_proj.dw1_norm(w1)  # BT22IN
        qkdd = self.dyn_w_proj.dw_activation(dd.view((B, T, 2, 2, N)))  # BT2{2}N1->BT2{2}N tanh
        qkw = torch.einsum('BTKJIN,BTKJIM->BTKJNM', w1, w2) + torch.diag_embed(qkdd)  # j=k=2, BT2{2}NM q/k, pre/post
        if self.query_wise:  # TODO: do not generate kw and kdd
            qw, _ = qkw.unbind(3)  # BS2NM
            kw_new = None
            qw = qw + self.dyn_w_proj.sw
        else:
            qw, kw_new = qkw.unbind(3)  # BS{pre/post}{q/k}NM -> BS{pre/post}NM * 2
            kw_new = kw_new + self.dyn_w_proj.sw  # BS2NM + 2NM-> BS2NM
        if self.kv_cache is not None:
            k, v, kw_out = self.kv_cache.update(input_pos, k, v, kw_val=kw_new)  # BNT2M
        logits = q @ k.transpose(-2, -1) * self.scale_factor
        if self.query_wise:
            w = qw  # B12NM
        else:
            w = qw + kw_out  # B12NM,BS2NM -> BS2NM
        wl, w = w.permute(0, 2, 3, 4, 1).unbind(1)  # BS2NM->B2NMS->[BNMS]*2
        logits = (logits * wl).sum(1).unsqueeze(2)  # BN1S, BNMS -> BNMS-> BMS-> BM1S
        min_value = torch.finfo(torch.float16).min
        logits = torch.where(k_mask, logits, min_value)
        probs = logits.softmax(-1)
        probs = (probs * w).sum(1).unsqueeze(2)
        y = probs @ v
        return y

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None,
                fast_infer=True) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)  # BSND
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))  # BNSD

        if self.is_training:
            N, D, I = self.n_head, self.head_dim, self.dyn_w_proj.dynamic_hidden_dim;  # 6.7B
            B, T, E = x.shape
            if self.use_dcmha:
                project_logits = True
                project_probs = True
                if project_probs:
                    dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2 * 2 * N * (2 * I), 2 * 2 * N * 1], -1)
                    dw_hidden = self.dyn_w_proj.dw_hidden_activation(dw_hidden)
                    dw_hidden = dw_hidden.view(dw_hidden.shape[:2] + (4, -1))  # B T (4 K) -> B T 4 K  # reshape
                    dw = torch.einsum('B T C K, C K D -> B T C D', dw_hidden,
                                      self.dyn_w_proj.qkw_m)  # BT4K,4K(MI)->BT4(MI)
                    shape = (B, T, 2 * 2, -1, N)  # if project_logits else (B,T,2,N,-1)  # BT(pre/post)(q/k)IN
                    w1, w2 = dw.view(shape).split(I, -2)
                    w1 = self.dyn_w_proj.dw1_norm(w1)  # BT22IN
                    if self.use_sw:
                        pre_sw, post_sw = self.dyn_w_proj.sw.unbind(0)
                    else:
                        pre_sw, post_sw = None, None
                    pre_qw1, pre_kw1, post_qw1, post_kw1 = w1.unbind(2)  # BT(2{*2})IN->[BTIN]*4
                    pre_qw2, pre_kw2, post_qw2, post_kw2 = w2.unbind(2)
                    qkdd = F.tanh(dd).squeeze(-1).view(shape[:-2] + (N,))  # BT(2{*2})N1->BT(2{*2})N
                    pre_qdd, pre_kdd, post_qdd, post_kdd = qkdd.unbind(2)  # BT(2{*2})N->[BTN]*4

                y = torch.zeros(B, N, T, D).to(q.device, dtype=torch.float16)
                for i in range(T // self.q_chunk_size):
                    start, stop = i * self.q_chunk_size, (i + 1) * self.q_chunk_size
                    kv_start = max(0, stop - self.q_chunk_size - self.window_size)
                    _q = q[:, :, start: stop, :]
                    _k, _v = k[:, :, kv_start: stop, :], v[:, :, kv_start: stop, :]
                    _atten_mask = mask[:, :, start: stop, kv_start: stop]
                    _pre_proj_dw_args = slice_dw(pre_sw, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd, start,
                                                 stop, kv_start) \
                        if project_logits else None
                    _post_proj_dw_args = slice_dw(post_sw, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd,
                                                  start, stop, kv_start) \
                        if project_probs else None
                    _o = _atten_context(_q, _k, _v, _atten_mask, _pre_proj_dw_args, _post_proj_dw_args)
                    y[:, :, start:stop] = _o
            else:
                y = torch.zeros(B, N, T, D).to(q.device, dtype=torch.float16)
                for i in range(T // self.q_chunk_size):
                    start, stop = i * self.q_chunk_size, (i + 1) * self.q_chunk_size
                    kv_start = max(0, stop - self.q_chunk_size - self.window_size)
                    _q = q[:, :, start: stop, :]
                    _k, _v = k[:, :, kv_start: stop, :], v[:, :, kv_start: stop, :]
                    _atten_mask = mask[:, :, start: stop, kv_start: stop]
                    _pre_proj_dw_args, _post_proj_dw_args = None, None
                    _o = _atten_context(_q, _k, _v, _atten_mask, _pre_proj_dw_args, _post_proj_dw_args)
                    y[:, :, start:stop] = _o
        else:  # inference
            if seqlen == 1:  # one-token generation
                k_mask = mask if self.window_size is None else mask[:, :, :, :self.kv_cache.seq_length]
                if fast_infer:
                    y = self._generate_fast(x, input_pos, q, k, v, k_mask)
                else:
                    assert not self.query_wise
                    # generate dw from hidden_state
                    pre_proj_dw_args, post_proj_dw_args, kw_new = self.dyn_w_proj(x, gen_cache=True)

                    # update kvkw cache
                    kw_new = kw_new + self.dyn_w_proj.sw  # absorb residual or sw into kw cache
                    if self.kv_cache is not None:
                        k, v, kw_out = self.kv_cache.update(input_pos, k, v, kw_val=kw_new)  # BNSD, BNSD, BS2NN

                    logits = q @ k.transpose(-2, -1) * self.scale_factor
                    # merge pre_w and apply it
                    pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
                    pre_qw = torch.einsum('BTGIN, BTGIM->BTNM', pre_qw1, pre_qw2) + torch.diag_embed(pre_qdd.squeeze(2))
                    pre_w = pre_qw + kw_out[:, :, 0]  # B1NM, BSNM -> BSNM
                    logits = self.dyn_w_proj.pre_proj(logits, proj_w=pre_w.squeeze(1))

                    logits = torch.where(k_mask, logits, torch.finfo(torch.float16).min)
                    probs = logits.softmax(-1)

                    # merge post_w and apply it
                    post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
                    post_qw = torch.einsum('BTGIN, BTGIM->BTNM', post_qw1, post_qw2) + torch.diag_embed(
                        post_qdd.squeeze(2))
                    post_w = post_qw + kw_out[:, :, 1]
                    probs = self.dyn_w_proj.post_proj(probs, proj_w=post_w.squeeze(1))

                    y = probs @ v
            else:  # prefill
                k_mask = mask[:, :, :, :k.shape[-2]]
                pre_proj_dw_args, post_proj_dw_args, kw_new = self.dyn_w_proj(x, gen_cache=True)
                kw_new = kw_new + self.dyn_w_proj.sw  # absorb residual or sw into kw cache
                if self.kv_cache is not None:
                    self.kv_cache.update(input_pos, k, v, kw_val=kw_new)  # BNSD, BNSD, BS2NN
                logits = q @ k.transpose(-2, -1) * self.scale_factor
                logits = self.dyn_w_proj.pre_proj(logits, dws=pre_proj_dw_args, query_vec=x, key_vec=x,
                                                  fast_infer=True)  # XD BN1S
                logits = torch.where(k_mask, logits, torch.finfo(torch.float16).min)
                probs = logits.softmax(-1)
                probs = self.dyn_w_proj.post_proj(probs, dws=post_proj_dw_args, query_vec=x, key_vec=x,
                                                  fast_infer=True)  # BN1S
                y = probs @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return y


class DynamicWeightProjection(nn.Module):

    def __init__(self, num_heads=32, num_groups=1, residual=True, query_input_dim=4096, dynamic_squeeze_ratio=16,
                 dynamic_w_hidden_dim=128, dtype=torch.float16, use_sw=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.query_input_dim = query_input_dim
        self.dynamic_squeeze_ratio = dynamic_squeeze_ratio
        self.dynamic_w_hidden_dim = dynamic_w_hidden_dim
        self.dw_hidden_activation = nn.GELU()
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.dw_activation = nn.Tanh()
        self.dw1_norm = RMSnormNoscale(dim=-1)
        self.use_sw = use_sw
        self.pre_proj = CrossHeadProjection('pre', num_heads=self.num_heads, use_sw=use_sw)
        self.post_proj = CrossHeadProjection('post', num_heads=self.num_heads, use_sw=use_sw)

        dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.dw1 = nn.parameter.Parameter(
            torch.zeros(self.query_input_dim, self.num_groups, 4, self.dynamic_w_hidden_dim,
                        dtype=dtype))  # (4096, 1, 4, 128)
        G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
        I = dynamic_hidden_dim * 2
        self.qkw = nn.parameter.Parameter(torch.zeros([G, 4, K, I, M], dtype=dtype))  # (1, 4, 128, 4, 32)
        self.dd = nn.parameter.Parameter(
            torch.zeros(self.query_input_dim, self.num_groups, self.num_heads_per_group * 4,
                        dtype=dtype))  # (4096, 1, 128)

        self.merge_weights()

    def merge_weights(self):
        self.dw_m = nn.parameter.Parameter(
            torch.cat([self.dw1.reshape(self.query_input_dim, -1), self.dd.squeeze(1)], dim=-1)).to(
            self.dw1.device)  # E,(4*K + K)  K=2*N*I
        self.qkw_m = nn.parameter.Parameter(
            self.qkw.permute(0, 1, 2, 3, 4).reshape(4, self.dynamic_w_hidden_dim, -1)).to(self.dw1.device)  # (4,K,I*M)
        if self.use_sw:
            self.sw = nn.parameter.Parameter(
                torch.stack([self.pre_proj.w, self.post_proj.w]).squeeze(1) + torch.eye(self.num_heads)).to(
                self.dw1.device)  # (2,N,N) sw + identity matrix
        else:
            self.sw = (torch.eye(self.num_heads).expand(2, self.num_heads, self.num_heads)).to(
                self.dw1.device)  # identity matrix (2,N,N)

    def forward(self, query_vec, KW: Optional[torch.Tensor] = None, gen_cache: Optional[bool] = True):
        dw_hidden = torch.einsum('BTD,DGCK->BTGCK', query_vec, self.dw1)  # C=4 [pre,post]*[query,key]
        dw_hidden = self.dw_hidden_activation(dw_hidden)  # BTGCK
        w1, w2 = torch.split(torch.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), self.qkw.shape[-2] // 2,
                             dim=-2)  # BTGC(2I)M -> [BTGCIM] * 2
        w1 = self.dw1_norm(w1)  # BTGCIM
        pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, dim=3)  # BTG4IM->[BTGIM]*4
        pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, dim=3)
        dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd)  # BTG(4M)
        dd = self.dw_activation(dd)
        pre_qdd, pre_kdd, post_qdd, post_kdd = torch.split(dd, dd.shape[-1] // 4, dim=-1)  # BTG(4N)->[BTGN]*4
        pre_dw_args = (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)
        post_dw_args = (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
        if gen_cache:  # generate KW cache
            pre_kw = torch.einsum('BSGIM, BSGIN->BSMN', pre_kw1, pre_kw2) + torch.diag_embed(
                pre_kdd.squeeze(2))  # merge kw and kdd
            post_kw = torch.einsum('BSGIM, BSGIN->BSMN', post_kw1, post_kw2) + torch.diag_embed(post_kdd.squeeze(2))
            KW = torch.stack((pre_kw, post_kw), dim=-3)  # BSMN,BSMN->BS2MN
        return pre_dw_args, post_dw_args, KW


class RMSnormNoscale(nn.Module):

    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs


class RMSnorm(nn.Module):

    def __init__(self, hid_dim=128, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.hid_dim = hid_dim
        self.epsilon = epsilon
        self.scale = nn.parameter.Parameter(data=torch.ones(self.hid_dim))

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs



def unbind(ary, n, dim=0):
    return [torch.squeeze(a, dim=dim) for a in torch.split(ary, ary.shape[dim] // n, dim=dim)]

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, mode='half') -> Tensor:
    if mode == 'half':
        xshaped = x.float().reshape(*x.shape[:-1], 2,-1).transpose(-1,-2)
    elif mode == 'alternative':
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def _atten_context(query, key, value, atten_mask, pre_proj_dw_args, post_proj_dw_args):
    logits = query @ key.transpose(-2, -1)
    if pre_proj_dw_args is not None: logits = _cross_head_proj(logits, *pre_proj_dw_args)
    logits = torch.where(atten_mask, logits, torch.finfo(torch.float16).min)
    probs = logits.softmax(-1)
    if post_proj_dw_args is not None: probs = _cross_head_proj(probs, *post_proj_dw_args)
    o = probs @ value  # BNTS,BNSD->BNTD
    return o



def _cross_head_proj(inputs, sw, qw1, qw2, kw1, kw2, qdd, kdd, loop_over_dynamic_hd=False):
    out = inputs + torch.einsum('BNTS,NM->BMTS', inputs, sw) if sw is not None else inputs
    for i in range(2): # qw1.shape[-2]):
        qhidden = (inputs * qw1[..., i, :].transpose(-2, -1).unsqueeze(-1)).sum(1)  # BNTS,(BTN->BNT->BNT1)->BNTS->BTS
        qout = qhidden.unsqueeze(1) * qw2[..., i, :].transpose(-2, -1).unsqueeze(-1) # (BTS->B1TS),(BTN->BNT->BNT1)->BNTS
        out = out + qout
        khidden = (inputs * kw1[..., i, :].transpose(-2, -1).unsqueeze(-2)).sum(1)  # BNTS,(BSN->BNS->BN1S)->BNTS->BTS
        kout = khidden.unsqueeze(1) * kw2[..., i, :].transpose(-2, -1).unsqueeze(-2) # (BTS->B1TS),(BSN->BNS->BNS1)->BNTS
        out = out + kout
    qdout = inputs * qdd.transpose(-2, -1).unsqueeze(-1); out = out + qdout  # BNTS,(BTN->BNT->BNT1)->BNTS
    kdout = inputs * kdd.transpose(-2, -1).unsqueeze(-2); out = out + kdout  # BNTS,(BSN->BNS->BN1S)->BNTS
    return out


def slice_dw(sw, qw1, qw2, kw1, kw2, qdd, kdd, start, stop, kv_start):
    return (sw,
            qw1[:, start : stop] if qw1 is not None else None,
            qw2[:, start : stop] if qw2 is not None else None,
            kw1[:, kv_start : stop] if kw1 is not None else None,
            kw2[:, kv_start : stop] if kw2 is not None else None,
            qdd[:, start : stop] if qdd is not None else None,
            kdd[:, kv_start : stop] if kdd is not None else None)


class CrossHeadProjection(nn.Module):

    def __init__(self, mode, num_heads=16, num_groups=1, dtype=torch.float16, use_sw=False):
        super().__init__()
        self.mode = mode
        self.use_sw = use_sw
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_heads_per_group = self.num_heads // self.num_groups
        if self.use_sw:
            self.w = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group, dtype=dtype))
        else:
            self.register_buffer('w', torch.eye(self.num_heads_per_group, dtype=dtype).expand(self.num_groups, self.num_heads_per_group, self.num_heads_per_group))

    def forward(self, inputs,
            dws:Optional[Tuple[Tensor,Tensor, Tensor,Tensor, Tensor,Tensor]]=None,
            query_vec=None, key_vec=None,
            proj_w:Optional[Tensor]=None,
            fast_infer=True):
        if proj_w is not None:
            ret = torch.einsum('BNTS,BSNM->BMTS', inputs, proj_w)
        else:
            assert dws is not None
            qw1, qw2, kw1, kw2, qdd, kdd = dws
            inputs = inputs.unsqueeze(1) #BNTS->BGNTS
            # apply sw
            ret = torch.einsum('BGMTS,GMN->BGNTS', inputs, self.w) if self.use_sw else inputs
            if fast_infer:
                inputs_label = 'BGMTS'
                hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I') # BGITS
                # apply qw and kw
                for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]):
                    dw_label = f'B{sym}G{hidden_sym}M'  # w1: BTGIM, dw_label:BTGIM
                    dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
                    eqn1 = f'{inputs_label},{dw_label}->{hidden_label}' # 'BGMTS,BTGMI->BGITS'
                    eqn2 = f'{hidden_label},{dw_label}->{inputs_label}' # 'BGITS,BTGMI->BGMTS'
                    for i in range(dynamic_hidden_dim):
                        hidden = torch.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :]) # BGMTS,BTG(I)M->BGTS
                        out = torch.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :]) #  'BG(I)TS,BTG(I)M->BGMTS'
                        ret = ret + out
                # apply qdd and kdd
                for sym, dd in zip(['T', 'S'], [qdd, kdd]):
                    dd_label = f'B{sym}GM'
                    dout = torch.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd) # BGMTS,B(T/S)GM->BGMTS
                    ret = ret + dout
            else:
                # apply qw and kw (BTGIN)
                x_inter = torch.einsum('BGNTS, BTGIN->BGTSI', inputs, qw1)
                qw_out = torch.einsum('BGTSI, BTGIN->BGNTS', x_inter, qw2)
                ret = ret + qw_out
                x_inter = torch.einsum('BGNTS, BSGIN->BGTSI', inputs, kw1)
                kw_out = torch.einsum('BGTSI, BSGIN->BGNTS', x_inter, kw2)
                ret = ret + kw_out

                # apply qdd(BTGN) and kdd(BSGN)
                ret = ret + torch.einsum('BGNTS, BTGN->BGNTS', inputs, qdd)
                ret = ret + torch.einsum('BGNTS, BSGN->BGNTS', inputs, kdd)
            ret = ret.squeeze(1) # BGNTS->BNTS
        return ret
