import tensorlayerx as tlx
from gammagl.layers.conv import HardGAO
from functools import partial


class HardGATModel(tlx.nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 k):
        super(HardGATModel, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = tlx.nn.ModuleList()
        self.activation = activation
        gat_layer = partial(HardGAO, k=k)
        muls = heads
        # input projection (no residual)
        self.gat_layers.append(gat_layer(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, concat=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(gat_layer(
                num_hidden*muls[l-1] , num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, concat=True))
        # output projection
        self.gat_layers.append(gat_layer(
            num_hidden*muls[-2] , num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, False, None, concat=False))

    def forward(self, x, edge_index, num_nodes):
        h = x
        for l in range(self.num_layers):
            h = self.gat_layers[l](h, edge_index, num_nodes)
        logits = self.gat_layers[-1](h, edge_index, num_nodes)
        return logits