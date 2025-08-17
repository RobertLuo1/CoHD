import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional, List
import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import functional as F
from einops import rearrange, repeat

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear
from detectron2.utils.registry import Registry

from .position_encoding import PositionEmbeddingSine
from .query_generation import Query_Embedding_Padding_Simple, Query_Updating_Block

from .aoc import AdaptiveObjectCounting
from .dha import Dynamic_Hierarchical_Selection
from .hsd import Semantic_Decoder

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module.
"""

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class NT_MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        h = [64, 128, 64]
        self.num_layers = len(h) + 1
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedReferringDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        rla_weight: float = 0.1,
        aux_loss: bool,
        weights: List[float],
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.aux_loss = aux_loss
        self.num_queries = num_queries
        self.weights = weights

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Hierarchical Semantic Decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_feature_levels = 3
        self.vis_hws = [30*30, 60*60, 120*120]
        Hierarchical_semantic_decoder = nn.ModuleList()
        self.query_update_block = nn.ModuleList()
        for idx in range(self.num_feature_levels):
            vis_hw = self.vis_hws[idx]
            semantic_decoder = Semantic_Decoder(vis_hw=vis_hw, num_q=num_queries, dim=hidden_dim)
                                    
            Hierarchical_semantic_decoder.append(semantic_decoder)
            if idx < 2:
                self.query_update_block.append(Query_Updating_Block(query_dim=hidden_dim))

        self.decoders = Hierarchical_semantic_decoder
        del Hierarchical_semantic_decoder

        ### define intra-and inter-level selection
        self.dha = Dynamic_Hierarchical_Selection()
       
       ### define counting head
        self.counting_head = AdaptiveObjectCounting(
            input_len=num_queries, input_dim=hidden_dim, hidden_dim=hidden_dim*2, num_classes=num_classes, num_layer=3)

        self.query_feat = Query_Embedding_Padding_Simple(
            out_query_num=num_queries, output_dim=hidden_dim, pre_merge=True, pos_type='fixed')

        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.nt_embed = NT_MLP(num_classes, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["aux_loss"] = cfg.MODEL.SEM_SEG_HEAD.AUX_LOSS
        ret["weights"] = cfg.MODEL.SEM_SEG_HEAD.WEIGHTS
        return ret

    def forward(self, x, mask_features, lang_feat, lang_sent=None, lang_mask=None):
        # x is a list of multi-scale feature
        # lang_feat: [B d l]
        # lang_sent: [B D]
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        x.append(mask_features)
        for i in range(self.num_feature_levels):
            size_list.append(x[i+1].shape[-2:])
            pos.append(self.pe_layer(x[i+1], None).flatten(2))
            src.append(self.input_proj[i](x[i+1]).flatten(2) + self.level_embed.weight[i][None, :, None])

        size_list.append(mask_features.shape[-2:])

        query_embed = None

        query = self.query_feat(lang_feat, lang_sent, lang_mask)                # [B Q d]
        
        outputs = []
        attns = []
        vis_pre = None
        output = None
        for idx, _decoder in enumerate(self.decoders):
            h, w = size_list[idx]
            output, attn, count, vis_pre = _decoder(query, src[idx], query_pos=query_embed, vis_pre=vis_pre)
            if idx < 2:
                output = self.query_update_block[idx](output, query)
                query = output

            attn = rearrange(attn, "b n q (h w) -> b n q h w", h=h, w=w) 
            attns.append(attn)
            outputs.append(output)
        
        prediction_masks = []

        pred_outputs = torch.stack(outputs)
        counting_pred, _ = self.counting_head(pred_outputs, None, aggregate=True)
        semantic_region_embeds = self.mask_embed(pred_outputs)
        class_embed = self.class_embed(semantic_region_embeds) #l b q a=2
        prediction_masks = self.dha(attns, outputs, class_embed, weights=self.weights)
        layer_outputs = []
        for pm in prediction_masks:
            layer_output = {
                "pred_masks": pm,
            }
            layer_outputs.append(layer_output)
        out = layer_outputs[-1]
        out["pred_count"] = counting_pred
        nt_feat = counting_pred.detach()
        nt_pred = self.nt_embed(nt_feat)
        out['nt_label'] = nt_pred
        if self.aux_loss:
            out["aux_outputs"] = layer_outputs[:-1]
        return out

    def semantic_inference(self, output, attns):
        """
        output: [b nq c]
        attns: [b nq h w]
        """
        #[b c h w]
        output = F.softmax(output, dim=-1)
        mask_pred = attns.sigmoid()
        mask_features = torch.einsum("bqc, bqhw -> bchw", output, mask_pred)
        return mask_features
    
