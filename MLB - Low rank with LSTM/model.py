import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
import counting


class Net(nn.Module):
    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2
        objects = 10

        self.text = TextProcessorWithAttention(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            attention_dim=512,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=glimpses,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=(glimpses * vision_features, question_features),
            mid_features=1024,
            out_features=config.max_answers,
            count_features=objects + 1,
            drop=0.5,
        )
        self.counter = counting.Counter(objects)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, b, q, q_len):
        q = self.text(q, list(q_len.data))

        v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v)

        a = self.attention(v, q)
        v = apply_attention(v, a)

        a1 = a[:, 0, :, :].contiguous().view(a.size(0), -1)
        count = self.counter(b, a1)

        answer = self.classifier(v, q, count)
        return answer


class LowRankMLBFusion(nn.Module):
    def __init__(self, x_features, y_features, z_features, rank=256, dropout=0.0):
        super(LowRankMLBFusion, self).__init__()
        self.linear_x1 = nn.Linear(x_features, rank)
        self.linear_y1 = nn.Linear(y_features, rank)
        self.linear_x2 = nn.Linear(rank, z_features)
        self.linear_y2 = nn.Linear(rank, z_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        x_proj = self.linear_x1(self.dropout(x))
        y_proj = self.linear_y1(self.dropout(y))
        combined = x_proj * y_proj
        output = self.linear_x2(combined) + self.linear_y2(combined)
        return output


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, count_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = LowRankMLBFusion(in_features[0], in_features[1], mid_features, rank=256, dropout=drop)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.lin_c = nn.Linear(count_features, mid_features)
        self.bn = nn.BatchNorm1d(mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)

    def forward(self, x, y, c):
        x = self.fusion(x, y)
        x = x + self.bn2(self.relu(self.lin_c(c)))
        x = self.lin2(self.drop(self.bn(x)))
        return x


class TextProcessorWithAttention(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, attention_dim, drop=0.0):
        super(TextProcessorWithAttention, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1,
                            batch_first=True)
        self.attention = nn.Linear(lstm_features, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def attention_net(self, lstm_output):
        attention_scores = self.tanh(self.attention(lstm_output))
        attention_weights = self.context_vector(attention_scores).squeeze(2)
        attention_weights = self.softmax(attention_weights)
        attention_weights = attention_weights.unsqueeze(2)
        context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        embedded = self.drop(embedded)

        packed = pack_padded_sequence(embedded, q_len, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        attn_output = self.attention_net(lstm_output)

        return attn_output


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = LowRankMLBFusion(mid_features * 100, mid_features * 100, mid_features * 100, rank=256,
                                       dropout=drop)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        v_flat = v.reshape(v.size(0), -1)
        q_flat = q.reshape(q.size(0), -1)
        x = self.fusion(v_flat, q_flat)
        x = x.reshape(v.size(0), -1, v.size(2), v.size(3))
        x = self.x_conv(self.drop(x))
        return x


def apply_attention(input, attention):
    n, c = input.size()[:2]
    glimpses = attention.size(1)
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)
    attention = attention.view(n * glimpses, -1)
    attention = F.softmax(attention, dim=1)
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    weighted_mean = weighted.sum(dim=3, keepdim=True)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    n, c = feature_vector.size()
    spatial_sizes = feature_map.size()[2:]
    tiled = feature_vector.view(n, c, *([1] * len(spatial_sizes))).expand(n, c, *spatial_sizes)
    return tiled
