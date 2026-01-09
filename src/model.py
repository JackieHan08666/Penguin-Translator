import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        nb = q.size(0)
        q_sl = q.size(1)
        k_sl = k.size(1)
        
        q = self.w_q(q).view(nb, q_sl, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(nb, k_sl, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(nb, k_sl, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # 兼容处理：将 mask 扩展为 (nb, 1, q_sl, k_sl)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            
            # 修复点：将 -1e9 改为 -1e4 以适配 FP16 范围
            # 或者使用更加动态的方法：
            fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, fill_value)
            
        p_attn = F.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(p_attn), v)
        
        x = x.transpose(1, 2).contiguous().view(nb, q_sl, self.h * self.d_k)
        return self.w_o(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        nx = self.norm1(x)
        x = x + self.dropout(self.self_attn(nx, nx, nx, mask))
        nx = self.norm2(x)
        x = x + self.dropout(self.feed_forward(nx))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.norm3 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, memory, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.src_attn(self.norm2(x), memory, memory, src_mask))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, n_layers=2, d_model=128, d_ff=512, n_heads=4, dropout=0.1, use_pe=True):
        super().__init__()
        self.use_pe = use_pe
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, MultiHeadAttention(d_model, n_heads, dropout), 
                             PositionwiseFeedForward(d_model, d_ff, dropout), dropout) 
            for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, MultiHeadAttention(d_model, n_heads, dropout), 
                         MultiHeadAttention(d_model, n_heads, dropout), 
                         PositionwiseFeedForward(d_model, d_ff, dropout), dropout) 
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.src_embed(src)
        memory = self.pe(src_emb) if self.use_pe else src_emb
        for layer in self.encoder_layers: 
            memory = layer(memory, src_mask)
        memory = self.encoder_norm(memory)
        
        tgt_emb = self.tgt_embed(tgt)
        x = self.pe(tgt_emb) if self.use_pe else tgt_emb
        for layer in self.decoder_layers: 
            x = layer(x, memory, src_mask, tgt_mask)
        return self.output_linear(x)