import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
from einops import rearrange
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class Dynamic_Hierarchical_Selection(torch.nn.Module):
    """
    Intra- and Inter-Selection on the object query
    """
    def __init__(self, levels=3, num_heads=8, hidden_dim=256, tokens=21):
        super().__init__()
        self.num_heads = num_heads
        self.level = levels
        self.tokens = tokens

        self.level_gating = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.token_gates = nn.ModuleList([
            ChannelAttention(in_channels=tokens, ratio=4) for _ in range(levels)
        ]
        )


    def forward(self, attn_maps: List[Tensor], outputs: List[Tensor], kernels: List[Tensor], weights: List[float]):
        # attn_maps: list[map=[B, n_h, N, H, W]]
        # kernels: List[kn=[B, N, 2]]
        # output: List[out=[B, N, C]]
        # return: List[mask=[B, 2, H, W]]

        semantic_features = [out.detach().mean(dim=1) for out in outputs]    # l * [B C]
        scores = self.level_gating(torch.stack(semantic_features))    # [l, B, 1]

        outputs_seg_masks = []
        prev_attn_mask = None
        cur_attn_mask = None

        for i_attn, attn in enumerate(attn_maps):
            attn = attn.sum(dim=1) / self.num_heads    # [B, N, H, W]

            token_score = self.token_gates[i_attn](attn)    # [B, N, 1, 1] 
            attn = attn * token_score.view(-1, self.tokens, 1, 1)    # token selection

            attn = attn * scores[i_attn].view(-1, 1, 1, 1)             # dynamic level weight

            prev_attn_mask = attn
            if cur_attn_mask is None:
                size = attn_maps[i_attn+1].size()[-2:]
                cur_attn_mask = F.interpolate(prev_attn_mask, size=size, mode='bilinear', align_corners=False)
                outputs_seg_masks.append(prev_attn_mask)
            else:
                cur_attn_mask =  weights[i_attn - 1] * cur_attn_mask + prev_attn_mask
                outputs_seg_masks.append(cur_attn_mask)
                if i_attn < len(attn_maps) - 1:
                    size = attn_maps[i_attn+1].size()[-2:]
                    cur_attn_mask = F.interpolate(cur_attn_mask, size=size, mode="bilinear", align_corners=False)

        prediction_masks = []
        for kn, pred_mask in zip(kernels, outputs_seg_masks):
            mask = torch.einsum("bqa, bqhw -> bahw", kn, pred_mask)
            prediction_masks.append(mask)
        return prediction_masks