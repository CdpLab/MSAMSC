from torch import einsum
from einops import rearrange, repeat
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from types import SimpleNamespace
from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np
import torch.nn.functional as F





def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d

#attention
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=8, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads  # h = 8 x = [2,50,128]
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h) # (B*h, 1, T2)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)



#feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
#cross_attention
class CrossSelfTransformer(nn.Module):
    def __init__(self, latent_dim, input_dim,  heads, dim_head, depth=1, ff_expansion=4, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(latent_dim, context_dim=input_dim, heads=8, dim_head=16, dropout=attn_dropout),
                Attention(latent_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(latent_dim, mult=ff_expansion, dropout=ff_dropout)
            ]))
        # Add an additional linear layer for adjusting the output shape of cross_attn
        self.adjust_cross_attn_output = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, context, mask=None, context_mask=None):
        for cross_attn, self_attn, ff in self.layers:
            # Get the original input x for residual connection
            x = cross_attn(x, context=context, mask=context_mask)
        return x

#cross_audio_attention
class CrossSelfAduioTransformer(nn.Module):
    def __init__(self, latent_dim, input_dim,  heads, dim_head, depth=1, ff_expansion=4, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(latent_dim, context_dim=input_dim, heads=8, dim_head=16, dropout=attn_dropout),
                Attention(latent_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(latent_dim, mult=ff_expansion, dropout=ff_dropout)
            ]))
        # Add an additional linear layer for adjusting the output shape of cross_attn
        self.adjust_cross_attn_output = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, context, mask=None, context_mask=None):
        for cross_attn, self_attn, ff in self.layers:
            # Get the original input x for residual connection
            x = cross_attn(x, context=context, mask=context_mask)
        return x


#门控单元判断相似性
import numpy as np
from sklearn.cross_decomposition import CCA

import numpy as np
from sklearn.cross_decomposition import CCA



class ModalSimilarityComparator:
    def __init__(self, threshold):
        self.threshold = threshold

    def compare(self, modal1, modal2):
        modal1_cpu = modal1.cpu().detach().numpy().reshape(-1)
        modal2_cpu = modal2.cpu().detach().numpy().reshape(-1)
        similarity = self.cosine_similarity(modal1_cpu, modal2_cpu)
        return similarity > self.threshold

    def cosine_similarity(self, vector1, vector2):
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        similarity = np.dot(vector1, vector2)
        return similarity
        


from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
import torch.nn as nn

def contrastive_loss(positive_pairs, negative_pairs, margin=0.5):
    # 检查正样本和负样本的长度
    len_pos = len(positive_pairs)
    len_neg = len(negative_pairs)

    # 如果长度不同，用最小长度来截断两者
    min_len = min(len_pos, len_neg)
    positive_pairs = positive_pairs[:min_len]
    negative_pairs = negative_pairs[:min_len]

    # 提取所有的正样本视频和文本张量
    positive_videos = torch.stack([pair[0] for pair in positive_pairs])
    positive_texts = torch.stack([pair[1] for pair in positive_pairs])

    # 计算它们之间的余弦相似性
    positive_sim = cosine_similarity(positive_videos, positive_texts, dim=1)

    # 现在计算negative_pairs的相似性
    negative_videos = torch.stack([pair[0] for pair in negative_pairs])
    negative_texts = torch.stack([pair[1] for pair in negative_pairs])
    negative_sim = cosine_similarity(negative_videos, negative_texts, dim=1)

    # 计算对比损失
    loss = F.relu(margin - positive_sim + negative_sim).mean()
    return loss





import random
#对比数据采样
def sample_pairs(videos, audios, texts, labels):
    labels = labels[1]   # 选择整数标签
    labels = labels.long()
    positive_pairs = []
    negative_pairs = []
    if isinstance(labels, torch.Tensor):
        labels = labels.long()
    else:
        labels = torch.tensor(labels, dtype=torch.long)

    # 确保labels是一个一维Tensor
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long) # 假设标签为整数，使用long类型
    elif len(labels.shape) > 1: # 如果labels是多维的，尝试减少其维度
        labels = labels.squeeze()

    for i in range(len(labels)):
        current_label = labels[i]
        
        # 获取与当前标签相匹配的所有索引
        same_label_indices = torch.nonzero(labels == current_label, as_tuple=True)[0]
        # 删除当前的索引，因为我们不想与其自身配对
        same_label_indices = same_label_indices[same_label_indices != i]
        # 如果有与当前标签相匹配的其他样本
        if len(same_label_indices) > 0:
            positive_video = videos[i]
            positive_text_idx = random.choice(same_label_indices).item()  # 随机选择一个
            positive_text = texts[positive_text_idx]
            positive_pairs.append((positive_video, positive_text))

        # 获取与当前标签不匹配的所有索引
        different_label_indices = torch.nonzero(labels != current_label, as_tuple=True)[0]
        if len(different_label_indices) > 0:
            negative_video = videos[i]
            negative_text_idx = random.choice(different_label_indices).item()  # 随机选择一个
            negative_text = texts[negative_text_idx]
            negative_pairs.append((negative_video, negative_text))

    return positive_pairs, negative_pairs



