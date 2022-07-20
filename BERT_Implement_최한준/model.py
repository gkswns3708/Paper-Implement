import torch
import torch.nn as nn
import torch.nn.functional as F


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # TODO : 아래에서 왜 seq_k에 i_pad를 붙이지?
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_hidn ** 0.5)
        
    def forward(self, Q, K, V, attn_mask):
        # scores shape : (bs, n_head,n_q_seq, n_k_seq) n_q_seq와 n_k_seq는 같은 값.
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        
        # attn_prob shape : (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(self.config.dropout)
        
        # context shape : (bs, n_head, n_q_seq, d_embedding)
        context = torch.matmul(attn_prob, V)
        
        return context, attn_prob
        
        

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config) 
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, Q, K, V, attn_mask):
        
        batch_size = Q.size(0)
        # q_s, k_s, v_s shape : (bs, n_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
        
        # context shape : (bs, n_head, n_q_seq, d_head)
        # attn_prob : (bs, n_head, n_q_seqn, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        
        # context shape : (bs, n_head, n_q_seq, n_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)
        
        # output shape : (bs, n_head, n_q_seq, e_embed)
        output = self.linear(context)
        output = self.dropout(output)
        
        return output, attn_prob
        
    
        


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, inputs):
        # output shape : (bs, n_seq, d_hidn)  -> (bs, d_ff, n_seq)
        output = self.active(self.conv1(inputs.transpose(1, 2)))  
        # output shape : (bs, d_ff, n_seq) -> (bs, n_seq, h_hidn)
        output = self.conv2(output).transpose(1, 2) # 1, 2임
        output = self.dropout(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, inputs, attn_mask):
        # attn_outputs shape : (bs, n_enc_seq, d_hidn)
        # attn_prob shape : (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.layer_norm1(inputs + attn_outputs) # Residual Connection
        
        # ffn_outputs shape : (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(attn_outputs)
        ffn_outputs = self.layer_norm2(attn_outputs + ffn_outputs)
        
        return ffn_outputs, attn_prob
        

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        self.pos_emb = nn.Embedding(self.config.n_enc_seq + 1, self.config.d_hidn) # [CLS] 토큰 때문에 + 1
        self.seg_emb = nn.Embedding(self.config.n_seg_type, self.config.d_hidn)
        
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
        
    def forward(self, inputs, segments):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.mask_fill_(pos_mask, 0)
        
        # outputs shape : (bs, n_enc_seq, d_hidn)
        outputs = self.enc_emb(inputs) + self.pos_emb(inputs) + self.seg_emb(inputs)
        
        # attn_mask shape : (bs, n_enc_seq, n_enc_seq)
        # get_attn_pad_mask : (input sequence + pad) * (input sequence + pad)로 된 matrix에서 padding부분의 attention을 지움.
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)
        
        attn_probs = []
        for layer in self.layers:
            # outputs shape : (bs, n_enc_seq, d_hidn)
            # attn_mask shape : (bs, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        
        return outputs, attn_probs
        
        



class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(self.config)
        self.linear = nn.Linear(config.d_hidn, config.d_hidn)
        self.activation = torch.tanh 
        
    def forward(self, inputs, segments):
        # output.shape : (bs, n_seq, d_hidn)
        # self_attn_probs.shape : (bs, n_head, n_enc_seq, n_enc_seq)
        outputs, self_attn_probs = self.encoder(inputs, segments)
        
        outputs_cls = outputs[:, 0].contiguous() 
        outputs_cls = self.linear(outputs_cls)
        outputs_cls = self.activation(outputs_cls)
        
        return outputs, outputs_cls, self_attn_probs
        
        
        


class BERTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.bert = BERT(self.config)