import torch.nn as nn
import torch
from einops import rearrange, repeat


class Query_Embedding_Padding_Simple(nn.Module):
    """ merge lang_feat and lang_sent"""

    def __init__(self, input_len=20, out_query_num=30, hidden_dim=768, 
                 output_dim=256, pre_merge=True, drop=0.0,
                 padding_type='sent', pos_type='fixed'):
        super().__init__()
        self.input_len = input_len
        self.out_query_num = out_query_num
        assert out_query_num >= input_len + 1
        learnable_num = out_query_num - 1 - input_len
        # assert learnable_num > 0
        self.learnable_num = learnable_num
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.pre_merge = pre_merge
        self.pos_type = pos_type
        assert pos_type in ['fixed', 'embed']

        self.padding_type = padding_type
        if pre_merge:
            if learnable_num > 0:
                self.learnable_query = nn.Embedding(self.learnable_num, self.hidden_dim)
            if pos_type == 'fixed':
                self.position_embed = FixedAbsolutePositionEmbedding(hidden_size=hidden_dim)
            else:
                self.position_embed = nn.Embedding(out_query_num, hidden_dim)
        else:
            if learnable_num > 0:
             self.learnable_query = nn.Embedding(self.learnable_num, self.output_dim)
            if pos_type == 'fixed':
                self.position_embed = FixedAbsolutePositionEmbedding(hidden_size=output_dim)
            else:
                self.position_embed = nn.Embedding(out_query_num, output_dim)

        if self.pre_merge:
            self.proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.feat_down = nn.Linear(hidden_dim, output_dim)
            self.sent_down = nn.Linear(hidden_dim, output_dim)
        self.out_drop = nn.Dropout(drop)

    def padding(self, lang_feat, lang_sent, lang_mask):
        """
        lang_feat: [B, Nl, D] = [B, 20, 768]
        lang_sent: [B, D] = [B, 768]
        lang_mask: [B, Nl, 1]
        return: [B l D] = [B, Q, 768]
        """
        b, l, d = lang_feat.size()
        lenth = torch.sum(lang_mask.squeeze(-1), dim=1)
        if self.padding_type == 'sent':
            for batch_id in range(b):
                lang_feat[batch_id, int(lenth[batch_id]):, ...] = lang_sent[batch_id]
        else:
            raise NotImplementedError
        return lang_feat
    
    def with_pos_embedding(self, feat):
        if self.pos_type == 'fixed':
            feat = self.position_embed(feat)
        else:
            feat = feat + self.position_embed.weight
        return feat

    def forward(self, lang_feat, lang_sent, lang_mask):
        """
        lang_feat: [B, D, Nl] = [B, 768, 20]
        lang_sent: [B, D] = [B, 768]
        lang_mask: [B, Nl, 1]
        return: [B Q d] = [B, Q, 256]
        """

        lang_feat = rearrange(lang_feat, "b d l -> b l d")
        b, l, d = lang_feat.size()
        lang_feat = self.padding(lang_feat, lang_sent, lang_mask)

        lang_sent = lang_sent.unsqueeze(1)
    
        if self.pre_merge:
            if self.learnable_num > 0:
                learn_query = self.learnable_query.weight.unsqueeze(0).expand((b, -1, -1))
                feat = torch.cat([lang_sent, lang_feat, learn_query], dim=1)    # [B 25 768]
            else:
                feat = torch.cat([lang_sent, lang_feat], dim=1)    # [B 21 768]
            feat = self.with_pos_embedding(feat)
            feat = self.out_drop(self.proj(feat))      # [B L 768] -> [B L 256]
        else:
            lang_sent = self.sent_down(lang_sent)
            lang_feat = self.feat_down(lang_feat)       # [..., 768] -> [..., 256]
            if self.learnable_num > 0:
                learn_query = self.learnable_query.weight.unsqueeze(0).expand((b, -1, -1))
                feat = torch.cat([lang_sent, lang_feat, learn_query], dim=1)    # [B 25 256]
            else:
                feat = torch.cat([lang_sent, lang_feat], dim=1)    # [B 25 256]
            feat = self.with_pos_embedding(feat)
            feat = self.out_drop(feat)

        return feat


class FixedAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings=64, hidden_size=256, position_embedding_type='rope'):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('embeddings', embeddings)

    def forward_fixed(self, x):
        """
        return (b l d)
        """
        return x + self.embeddings[None, :x.size(1), :]

    def forward_rope(self, x):
        """
        return (b l d)
        """
        embeddings = self.embeddings[None, :x.size(1), :] # b l d
        embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
        sin, cos = embeddings.unbind(dim=-2) # b l d//2
        sin, cos = map(lambda t: repeat(t, '... d -> ... (d 2)'), (sin, cos)) # b l d
        return x * cos + self.rotate_every_two(x) * sin

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d j -> ... (d j)')

    def _forward(self, x):
        if self.position_embedding_type == 'fixed':
            return self.forward_fixed(x)

        elif self.position_embedding_type == 'rope':
            return self.forward_rope(x)

    def forward(self, x):
        if x.dim() == 3:
            return self._forward(x)

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        # nn.init.normal_(m.weight, std=0.02)
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    else:
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class Query_Updating_Block(nn.Module):
    def __init__(self, query_dim, num_heads=8, drop=0.15):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(query_dim, num_heads=num_heads)
        # self.attn2 = nn.MultiheadAttention(query_dim, num_heads=num_heads)

        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(query_dim)
    
    def forward(self, decoder_output, prev_output):
        """
        decoder_output: [B Q C]
        prev_output: [B Q C]
        """
        prev_output = rearrange(prev_output, "b q c -> q b c")

        decoder_output = rearrange(decoder_output, "b q c -> q b c")

        out1 = self.attn1(query=decoder_output, key=prev_output, value=prev_output)[0]

        decoder_output = decoder_output + self.dropout(out1)

        decoder_output = self.norm(decoder_output)

        decoder_output = rearrange(decoder_output, "q b c -> b q c")

        return decoder_output