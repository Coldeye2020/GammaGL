import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax, degree
import math
import numpy as np


class Identity(tlx.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class HardGAO(MessagePassing):
    def __init__(self,
                in_channels,
                out_channels,
                heads=1,
                feat_drop=0,
                attn_drop=0,
                negative_slope=0.2,
                residual=True,
                activation=tlx.elu,
                k=8,
                concat=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negetive_slop = negative_slope
        self.residual = residual
        self.k = k
        self.concat = concat

        # Initialize Parameters for Additive Attention
        self.linear = tlx.layers.Linear(self.in_feats, self.out_feats * self.heads, bias=False)

        initor = tlx.initializers.XavierNormal(gain=math.sqrt(2))
        self.att_src = self._get_weights("att_src", shape=(1, self.heads, self.out_channels), init=initor, order=True)
        self.att_dst = self._get_weights("att_dst", shape=(1, self.heads, self.out_channels), init=initor, order=True)

        # Initialize Parameters for Hard Projection
        self.p = self._get_weights("proj", shape=(1,self.in_channels), init=initor, order=True)
        # Initialize Dropouts
        self.feat_drop = tlx.layers.Dropout(feat_drop)
        self.feat_drop = tlx.layers.Dropout(attn_drop)
        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)
        if self.residual:
            if self.in_feats == self.out_feats:
                self.residual_module = Identity()
            else:
                self.residual_module = tlx.layers.Linear(self.in_feats,self.out_feats*self.heads,bias=False)
        self.activation = activation

    def select_topk(edge_index, value, k=2):
        src, dst = tlx.convert_to_numpy(edge_index)
        score = tlx.convert_to_numpy(value)
        score = score[src]

        # 根据value进行降序排列
        rank = np.argsort(score)[::-1]
        src = src[rank]
        dst = dst[rank]
        # 根据dst进行升序排列
        index = np.argsort(dst, kind='mergesort')
        src = src[index]
        dst = dst[index]
        
        # 直接用lexsort好像一步就可以，但是可读性较差
        # index = np.lexsort((-score, dst))
        # src = src[index]
        # dst = dst[index]

        # 每个dst节点筛选前k个边
        e_id = []
        rowptr = np.concatenate(([0], np.bincount(dst).cumsum()))
        for dst_node in np.unique(dst):
            st = rowptr[dst_node]
            len = rowptr[dst_node + 1] - st
            for i in range(st, min(st + k, st + len)):
                e_id.append(i)
        
        return tlx.convert_to_tensor([src[e_id], dst[e_id]])
        

    def message(self, x, edge_index, edge_weight=None, num_nodes=None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        weight_src = tlx.gather(tlx.reduce_sum(x * self.att_src, -1), node_src)
        weight_dst = tlx.gather(tlx.reduce_sum(x * self.att_dst, -1), node_dst)
        weight = self.leaky_relu(weight_src + weight_dst)

        # el = (x * self.attn_l).sum(dim=-1).unsqueeze(-1)
        # er = (x * self.attn_r).sum(dim=-1).unsqueeze(-1)

        alpha = self.dropout(segment_softmax(weight, node_dst, num_nodes))
        x = tlx.gather(x, node_src) * tlx.expand_dims(alpha, -1)
        return x * edge_weight if edge_weight else x
    
    def forward(self, x, edge_index, num_nodes):
        # 这里如果以后实现了indegree最好替换一下
        if (degree(edge_index[1], num_nodes) == 0).any():
            raise NotImplementedError('There are 0-in-degree nodes in the graph, '
                                    'output for those nodes will be invalid. '
                                    'This is harmful for some applications, '
                                    'causing silent performance regression. ')
        
        # projection process to get importance vector y
        # graph.ndata['y'] = torch.abs(torch.matmul(self.p,feat.T).view(-1))/torch.norm(self.p,p=2)
        y = tlx.abs(tlx.squeeze(tlx.matmul(self.p, tlx.transpose(x))))/tlx.ops.l2_normalize(self.p)
        # Use edge message passing function to get the weight from src node
        # graph.apply_edges(fn.copy_u('y','y'))
        # Select Top k neighbors 生成新的图
        edge_index = self.select_topk(edge_index, y, self.k)
        # Sigmoid as information threshold
        y = tlx.sigmoid(y)
        # Using vector matrix elementwise mul for acceleration
        x = tlx.reshape(y, shape=(-1, 1)) * x
        x = self.feat_drop(x)
        h = tlx.reshape(self.linear(x), size=(-1, self.heads, self.out_channels))
        rst = self.propagate(h, edge_index, num_nodes=num_nodes)
        # activation
        if self.activation:
            rst = self.activation(rst)
        # Residual
        if self.residual:
            rst = rst + tlx.reshape(self.residual_module(x), shape=(x.shape[0], -1, self.out_feats))
        
        if self.concat:
            return tlx.reshape(rst, (-1, self.heads * self.out_channels))
        else:
            return tlx.reduce_mean(rst, axis=1)




            
        


