import math
import warnings

import torch
import torch.nn as nn

from utils.masking import Masking


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, model_name="CIRCALLM"):
        super(PositionalEmbedding, self).__init__()
        self.model_name = model_name

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)#0::2 表示从第0列开始，每隔2列选择一列，即选择所有偶数索引列（0, 2, 4, ...）。
        pe[:, 1::2] = torch.cos(position * div_term)#1::2 表示从第1列开始，每隔2列选择一列，即选择所有奇数索引列（1, 3, 5, ...）。

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if (
            self.model_name == "CIRCALLM"
            or self.model_name == "TimesNet"
            or self.model_name == "Moment"
            or self.model_name == "GPT4TS"
        ):
            return self.pe[:, : x.size(2)]#取pe的前x.size(2)列，x.size(2)表示被分成了多少个patch
        else:
            return self.pe[:, : x.size(1)]#FAN


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x = x.transpose(1, 2)
        # batch_size x seq_len x d_model
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
        self, c_in, d_model, model_name, embed_type="fixed", freq="h", patch_dropout=0.1
    ):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(
            d_model=d_model, model_name=model_name
        )
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=patch_dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", patch_dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=patch_dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
class AttentionChannelEmbedding(nn.Module):
    def __init__(self, d_model):
        super(AttentionChannelEmbedding, self).__init__()
        self.d_model = d_model
        # Attention mechanism
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, c_in]
        batch_size, seq_len, c_in = x.shape
        x_flattened = x.view(batch_size * seq_len, c_in)  # [batch_size * seq_len, c_in]

        # Query-Key-Value attention
        query = self.query(x_flattened)
        key = self.key(x_flattened)
        value = self.value(x_flattened)

        attention_weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)), dim=-1)
        embedded = torch.matmul(attention_weights, value)  # [batch_size * seq_len, d_model]

        return embedded.view(batch_size, seq_len, self.d_model)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        seq_len: int = 72,
        patch_len: int = 6,
        stride: int = 6,
        patch_dropout: int = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding
        #in_features：输入张量的特征数（即输入的维度）。out_features：输出张量的特征数（即输出的维度）。
        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))#nn.Parameter 是 PyTorch 中一个用于 标记张量为模型可学习参数 的类

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)#初始化权重矩阵
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()
            # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails

        # Positional embedding
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(patch_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mask = Masking.convert_seq_to_patch_view(
            mask, patch_len=self.patch_len
        ).unsqueeze(-1)
        # mask : [batch_size x n_patches x 1]
        n_channels = x.shape[1]
        mask = (
            mask.repeat_interleave(self.d_model, dim=-1)
            .unsqueeze(1)
            .repeat(1, n_channels, 1, 1)
        )
        # mask : [batch_size x n_channels x n_patches x d_model]

        # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


class CircaTokenEmbedding(nn.Module):
    def __init__(self, d_model: int = 768,bias: bool = False):
        super(CircaTokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        # 定义一个共享的 Conv1d，用于对每个通道独立操作
        self.tokenConv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=padding, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        """
        输入: x [batch_size, n_channels, seq_len]
        输出: [batch_size, n_channels, d_model]
        """ 
        # 将每个通道分开，逐通道独立应用 Conv1d
        channels = torch.unbind(x, dim=1)  # 拆分成 n_channels 个 [batch_size, 1, seq_len]
        embedded_channels = [
            self.tokenConv(channel.unsqueeze(1)).transpose(1, 2) for channel in channels
        ]  # 每个通道变成 [batch_size, d_model]
        # 合并所有通道
        return torch.stack(embedded_channels, dim=1)  # 最终形状 [batch_size, n_channels, d_model]

class CircaTemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed"):
        super(CircaTemporalEmbedding, self).__init__()
        minute_size = 60
        hour_size = 72
        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model)#0,1,2,3,59
        self.hour_embed = Embed(hour_size, d_model)#0,1,2,3...,71

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 1])
        hour_x = self.hour_embed(x[:, :, 0])

        return hour_x + minute_x
    

class CircaDataEmbedding(nn.Module):
    def __init__(
        self, 
        d_model: int = 768,
        patch_dropout=0.1,
        embed_type="fixed", 
        freq="h", 
        value_embedding_bias: bool = False,
        add_positional_embedding: bool = True,
        orth_gain: float = 1.41,
    ):
        super(CircaDataEmbedding, self).__init__()
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding

        self.mask_embedding = nn.Parameter(torch.zeros(self.d_model))#nn.Parameter 是 PyTorch 中一个用于 标记张量为模型可学习参数 的类

        #时间序列embedding
        self.value_embedding=CircaTokenEmbedding(d_model=self.d_model,bias=value_embedding_bias)
        
        #（位置+时间）编码
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(self.d_model)
            self.temporal_embedding = (CircaTemporalEmbedding(d_model=self.d_model, embed_type=embed_type) if embed_type != "timeF" 
                                    else TimeFeatureEmbedding(d_model=self.d_model, embed_type=embed_type, freq=freq))
        
        self.dropout = nn.Dropout(p=patch_dropout)

    def forward(self, x:torch.Tensor, mask:torch.Tensor = None,x_mark:torch.Tensor = None)->torch.Tensor:
        n_channels = x.shape[1]
        mask = mask.unsqueeze(-1)
        mask = (
            mask.repeat_interleave(self.d_model, dim=-1)
            .unsqueeze(1)
            .repeat(1, n_channels, 1, 1)
        )

        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x)
            if x_mark is not None:
                x = x+self.temporal_embedding(x_mark).unsqueeze(1).repeat(1, n_channels, 1, 1)

        return self.dropout(x)

class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if self.stride != self.patch_len:
            warnings.warn(
                "Stride and patch length are not equal. "
                "This may lead to unexpected behavior."
            )

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x : [batch_size x n_channels x num_patch x patch_len]
        return x
