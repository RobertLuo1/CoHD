import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F
import math

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
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


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv, save_attn=False):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if save_attn:
            attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if save_attn:
            return x, attn_save
        else:
            return x

class SDM_Attention(nn.Module):
    """
    Attention utilization for Semantic Decoder
    """
    def __init__(self, input_dim, hw, num_q, hidden_dim, 
                 num_head=None, qkv_bias=True, attn_drop=0.1, proj_drop=0.2, vis_update=False) -> None:
        super().__init__()
        self.num_head = num_head
        head_dim = input_dim // num_head
        self.scale = head_dim ** -0.5

        self.vis_k = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.vis_v = nn.Linear(input_dim, input_dim, bias=qkv_bias)

        self.lang_k = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.lang_v = nn.Linear(input_dim, input_dim, bias=qkv_bias)

        self.hw = hw
        self.out_proj = MLP(hw, hidden_dim, input_dim, 2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.vis_update = vis_update
        if vis_update:
            self.vis_update = nn.Conv1d(num_q, input_dim, 1)
            self.vis_update_drop = nn.Dropout(proj_drop)


    def forward(self, vis_feat, lang_feat, return_heads=False):
        """
        vis_feat: [B HW C]
        lang_feat: [B Q C]
        return: 
        """
        b_, hw_, c_ = vis_feat.shape
        _, q_, _ = lang_feat.shape

        v_k = self.vis_k(vis_feat).reshape(b_, hw_, self.num_head,
                                      c_ // self.num_head).permute(0, 2, 1, 3)
        v_v = self.vis_v(vis_feat).reshape(b_, hw_, self.num_head,
                                      c_ // self.num_head).permute(0, 2, 1, 3)

        l_k = self.lang_k(lang_feat).reshape(b_, q_, self.num_head,
                                      c_ // self.num_head).permute(0, 2, 1, 3)
        l_v = self.lang_v(lang_feat).reshape(b_, q_, self.num_head,
                                      c_ // self.num_head).permute(0, 2, 1, 3)

        A_mul = (l_k @ v_k.transpose(-2, -1)) * self.scale   # [B h Q HW]
        
        v_a = A_mul.softmax(dim=-1) @ v_v                    # [B h Q C/h]
        l_a = A_mul.transpose(-2, -1).softmax(dim=-1) @ l_v  # [B h HW C/h]

        f_mul = v_a @ l_a.transpose(-2, -1)         # [B h Q HW]
        attn_mask = f_mul.clone()

        f_mul = self.attn_drop(f_mul)
        f_mul = f_mul.mean(dim=1)                   # [B Q HW]
        out = self.out_proj(f_mul)                  # [B Q C]
        out = self.proj_drop(out)

        if self.vis_update:
            vis_ = f_mul.clone()
            vis_update = F.tanh(self.vis_update(vis_).transpose(-2, -1))     # [B HW C]
            vis_update = self.vis_update_drop(vis_update)

        if not return_heads:
            attn_mask = attn_mask.sum(dim=1) / self.num_head
        
        return out, attn_mask, vis_update if self.vis_update else None
    

class Semantic_Decoder_Module(nn.Module):
    def __init__(self, input_dim, vis_hw, num_q, num_head, hidden_dim, drop=0.2, vis_update=False) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.vis_hw = vis_hw
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.drop_p = drop
        self.vis_update = vis_update

        self.sdm_att = SDM_Attention(input_dim, vis_hw, num_q, hidden_dim, num_head, vis_update=vis_update)
        self.drop2 = nn.Dropout(0.0)
        self.norm2 = nn.LayerNorm(input_dim)

        self.cross_att = Attention(input_dim, num_head)
        self.drop3 = nn.Dropout(drop)
        self.norm3 = nn.LayerNorm(input_dim)

        self.out_proj = nn.Linear(input_dim, input_dim)
        self.norm4 = nn.LayerNorm(input_dim)


    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, query, vis_feat, query_pos=None, return_heads=False):
        """
        query: [B Q C]
        vis_feat: [B HW C]
        return: [B Q C], [B Q HW]
        """        
        out = self.with_pos_embed(query, query_pos)

        sdm_out, attn_mask, vis_update = self.sdm_att(vis_feat, out, return_heads)  # [B Q C], [B Q HW]
        sdm_out = self.norm2(out + self.drop2(sdm_out))

        out2 = self.cross_att(out, sdm_out, sdm_out)      # [B Q C]
        out2 = self.norm3(out + self.drop3(out2))

        out3 = self.out_proj(out2)
        out3 = self.norm4(out2 + F.dropout(out3, p=self.drop_p, training=self.training))
        out = F.gelu(out3)
        attn_count = out

        return out, attn_mask, attn_count, vis_update
    

class Semantic_Decoder(nn.Module):
    def __init__(self, vis_hw, num_q, layer_num=3, dim=256, hidden_dim=512, num_head=8,
                 vis_update=False, return_heads=True, vis_connect=False, get_pre_vis=False,
                 get_pre_output=False):
        super().__init__()
        self.layer_num = layer_num
        self.vis_hw = vis_hw
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        
        self.return_heads = return_heads
        self.vis_connect = vis_connect          # give a vis_update for next decoder
        self.get_pre_vis = get_pre_vis          # get a vis_update from uper decoder
        self.get_pre_output = get_pre_output    # get output from uper decoder

        if get_pre_vis:
            self.vis_expand = PatchExpand(dim)
            self.vis_trans = PatchTransition(dim + dim // 2, dim)

        if get_pre_output:
            self.query_trans = PatchTransition(dim * 2, dim)

        self.layers = nn.ModuleList()
        self.vis_updates = [vis_update] * (layer_num - 1) + [vis_connect]
        for i in range(layer_num):
            layer = Semantic_Decoder_Module(dim, vis_hw, num_q, num_head, hidden_dim, vis_update=self.vis_updates[i])
            self.layers.append(layer)

    def forward(self, query, vis_feat, query_pos=None, output_pre=None, vis_pre=None):
        """
        query: [B Q C]
        vis_feat: [B C HW]
        vis_pre: None or [B hw C]
        return: [B Q C], [B Q HW]
        """
        output = query
        vis_feat = rearrange(vis_feat, 'b c n -> b n c')        # to [B HW C]

        if self.get_pre_vis:
            assert vis_pre is not None
            vis_pre = self.vis_expand(vis_pre)      # [B hw C] -> [B HW C//2]
            vis_feat = self.vis_trans(torch.cat([vis_feat, vis_pre], dim=-1)) # [B HW C]
        
        if self.get_pre_output:
            assert output_pre is not None
            output = self.query_trans(torch.cat([output, output_pre], dim=-1))

        for idx, layer in enumerate(self.layers):
            output, attn_mask, attn_count, vis_update = layer(output, vis_feat, query_pos, self.return_heads)
            if self.vis_updates[idx]:
                vis_feat = vis_feat + vis_update
            
        return output, attn_mask, attn_count, vis_feat if self.vis_connect else None

class PatchExpand(nn.Module):
    """
    Pacth Expand Operation
    """
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        """
        Args:
            input_resolution: the height and width of the feature map
            dim: the channel number of the feature map
            dim_scale: the scale of the channel number
            norm_layer: the normalization layer
        """
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, L, C = x.shape
        H = W = int(math.sqrt(L))

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x

class PatchTransition(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.trans = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        """
        x: [B HW C2] or List[ [B HW C] ]
        """
        if isinstance(x, torch.Tensor):
            pass
        elif isinstance(x, list):
            x = torch.cat(x, dim=-1)
            assert x.size[-1] == self.input_dim
        else:
            raise TypeError
        x = self.trans(x)
        x = self.norm(x)
        return x

if __name__ == "__main__":
    Net = Semantic_Decoder(900)
    vis = torch.randn((2, 900, 768))
    lang = torch.randn((2, 40, 768))
    out, attn_mask = Net(lang, vis)
    print('ok')
