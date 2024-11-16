import math
import torch
from torch import nn


def fps_torch(pos: torch.tensor, k: int = 1, select: int = 0) -> torch.tensor:
    """Taken from previous project.
    Args:
        pos (torch.tensor): (N, 3) tensor of positions
        k (int): number of samples to select
        select (int, optional): Optional initial position to be sampled

    Return:
        torch.tensor: (k, 3) tensor of sampled positions
    """

    n = pos.shape[0]
    assert pos.shape[1] == 3, "Position tensor must be (N, 3)"
    assert (
        k <= n
    ), "Number of samples must be less than or equal to the number of positions"

    selected = torch.zeros(n, dtype=torch.bool)
    selected[select] = True

    distances = torch.pairwise_distance(pos, pos[selected])
    distances[selected] = -math.inf

    for _ in range(k - 1):
        new_sample = torch.argmax(distances)
        selected[new_sample] = True
        distances = torch.min(
            distances, torch.pairwise_distance(pos, pos[new_sample])
        )
        distances[selected] = -math.inf

    return pos[selected]


class HierarchicalFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(HierarchicalFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.linear = nn.Linear(self._cfg.c_s, self._cfg.c_s)

    def forward(self, h, pos):
        # [b, n_res, c_s]
        _, num_res, _ = h.shape
        n_nodes = (num_res * self._cfg.n_p) // 1
        new_node_pos = fps_torch(pos, n_nodes)
        dist = torch.cdist(pos, new_node_pos, p=3)  # [num_res, n_nodes]
        # assign edges between layers based on distance
        new_map = torch.nonzero(dist < self._cfg.cutoff, as_tuple=False)
        new_node_embed = self.linear(new_map.T @ h)

        return new_node_embed, new_node_pos, new_map
