import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.hierarchical_feature_net import HierarchicalFeatureNet
from models import ipa_pytorch
from data import utils as du


class HierarchicalFlowModel(nn.Module):

    def __init__(self, model_conf):
        super(HierarchicalFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self._hier_conf = model_conf.hier
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE
        )
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE
        )
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        self.hierarchical_feature_net = HierarchicalFeatureNet(
            model_conf.hier_features
        )

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = ipa_pytorch.InvariantPointAttention(
                self._ipa_conf
            )
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer,
                self._ipa_conf.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final"
            )
            self.trunk[f"node_transition_{b}"] = (
                ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            )
            self.trunk[f"bb_update_{b}"] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True
            )

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = (
                    ipa_pytorch.EdgeTransition(
                        node_embed_size=self._ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )
                )

        # Hierarchical trunk
        self.hierarchy = nn.ModuleDict()
        for i in range(self._hier_conf.num_layers):
            self.hierarchy[f"hier_{i}_update"] = ipa_pytorch.Linear(
                self._hier_conf.c_s, self._hier_conf.c_s
            )

    def forward(self, input_feats):
        node_mask = input_feats["res_mask"]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats["diffuse_mask"]
        res_index = input_feats["res_idx"]
        so3_t = input_feats["so3_t"]
        r3_t = input_feats["r3_t"]
        trans_t = input_feats["trans_t"]
        rotmats_t = input_feats["rotmats_t"]

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t, r3_t, node_mask, diffuse_mask, res_index
        )
        if "trans_sc" not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats["trans_sc"]

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        # Initialize hierarchical node embeddings
        hier_node_embed = [init_node_embed]
        hier_node_pos = [trans_t]
        hier_map = []  # list of mapping matrices from one layer to the next
        for _ in range(self._hier_conf.num_layers):
            new_node_embed, new_node_pos, new_map = (
                self.hierarchical_feature_net(
                    hier_node_embed[-1], hier_node_pos[-1]
                )
            )
            hier_node_embed.append(new_node_embed)
            hier_node_pos.append(new_node_pos)
            hier_map.append(new_map)

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            # same as before: aggregation
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed, edge_embed, curr_rigids, node_mask
            )
            ipa_embed *= node_mask[..., None]
            # Run aggregation and update across layers
            for i in range(0, self._hier_conf.num_layers):
                # aggregate above, below, and within current (probabl change)
                hier_node_embed[i] = self.trunk[f"hier_{i}_update"](
                    (hier_map[i - 1].T @ hier_node_embed[i - 1])
                    + (hier_map[i] @ hier_node_embed[i + 1])
                    + hier_node_embed[i]
                )
            # update
            node_embed = self.trunk[f"ipa_ln_{b}"](
                node_embed + ipa_embed + hier_node_embed[0]
            )
            # same as before: update w/ transformer
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](
                seq_tfmr_out
            )
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](
                node_embed * node_mask[..., None]
            )
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None]
            )
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](
                    node_embed, edge_embed
                )
                edge_embed *= edge_mask[..., None]
        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return {
            "pred_trans": pred_trans,
            "pred_rotmats": pred_rotmats,
        }
