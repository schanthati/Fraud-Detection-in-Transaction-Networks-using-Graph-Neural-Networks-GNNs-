# Fraud-Detection-in-Transaction-Networks-using-Graph-Neural-Networks-GNNs-

# NOTE: Make sure to install PyTorch and PyTorch Geometric
# pip install torch torch_geometric

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import random

# ----------------------------
# Step 1: Simulate Transaction Graph
# ----------------------------
num_nodes = 100
num_edges = 500

# Node features: random (e.g., activity, age, etc.)
x = torch.rand((num_nodes, 5))

# Generate random edges (src, dst)
edges = []
while len(edges) < num_edges:
    src = random.randint(0, num_nodes - 1)
    dst = random.randint(0, num_nodes - 1)
    if src != dst:
        edges.append((src, dst))
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Edge features: amount (0-1), time (0-1), location (one-hot for 5 locations)
amounts = torch.rand(num_edges, 1)
times = torch.rand(num_edges, 1)
locations = F.one_hot(torch.randint(0, 5, (num_edges,)), num_classes=5).float()
edge_attr = torch.cat([amounts, times, locations], dim=1)

# Labels: 10% are fraud (class 1)
labels = torch.zeros(num_edges)
fraud_indices = torch.randperm(num_edges)[:int(0.1 * num_edges)]
labels[fraud_indices] = 1

# Construct PyG data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_label=labels)

# ----------------------------
# Step 2: Define the GNN Model
# ----------------------------
class FraudGNN(torch.nn.Module):
    def __init__(self, in_node_feats, hidden_feats, edge_feat_dim):
        super().__init__()
        self.gnn = GATConv(in_node_feats, hidden_feats)
        self.edge_mlp = Linear(hidden_feats * 2 + edge_feat_dim, 1)

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index)
        src, dst = data.edge_index
        edge_repr = torch.cat([x[src], x[dst], data.edge_attr], dim=1)
        return torch.sigmoid(self.edge_mlp(edge_repr)).squeeze()

# ----------------------------
# Step 3: Train the Model
# ----------------------------
model = FraudGNN(in_node_feats=5, hidden_feats=16, edge_feat_dim=7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

# Train/test split
edge_idx = torch.randperm(data.edge_index.shape[1])
train_mask = edge_idx[:int(0.8 * len(edge_idx))]
test_mask = edge_idx[int(0.8 * len(edge_idx)):]

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[train_mask], data.edge_label[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            pred = (out[test_mask] > 0.5).float()
            acc = (pred == data.edge_label[test_mask]).float().mean()
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")
