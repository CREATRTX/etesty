inputs['agent_node_masks']['pedestrians']), dim = 2)
att_op, _ = self.a_n_att(queries, keys, vals, attn_mask=attn_masks)  # MultiheadAttention
att_op = att_op.permute(1, 0, 2)

# Concatenate with original node encodings and 1x1 conv
lane_node_enc = self.leaky_relu(self.mix(torch.cat((lane_node_enc, att_op), dim=2)))

# GAT layers    先构建邻接矩阵
adj_mat = self.build_adj_mat(inputs['map_representation']['s_next'], inputs['map_representation']['edge_type'])
for gat_layer in self.gat:  # 循环叠加矩阵
    lane_node_enc += gat_layer(lane_node_enc, adj_mat)

# Lane node masks
lane_node_masks = ~lane_node_masks[:, :, :, 0].bool()
lane_node_masks = lane_node_masks.any(dim=2)  # 全为0，返回FLASE 否则TRUE
lane_node_masks = ~lane_node_masks
lane_node_masks = lane_node_masks.float()

# Return encodings
# 目标编码+周围编码
encodings = {'target_agent_encoding': target_agent_enc,
             'context_encoding': {'combined': lane_node_enc,
'combined_masks': lane_node_masks,
'map': None,
'vehicles': None,
'pedestrians': None,
'map_masks': None,
'vehicle_masks': None,
'pedestrian_masks': None
},