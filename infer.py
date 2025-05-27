import torch
import random
from models.gnn_model import GNNLinkPredictor
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage

# Add PyTorch Geometric classes to safe globals list
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage])

# === Load model & data ===
data = torch.load('output/graph/hetero_graph.pt')
metadata = data.metadata()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNNLinkPredictor(hidden_channels=64, metadata=metadata).to(device)
model.load_state_dict(torch.load('output/model/best_gnn_model.pth'))
model.eval()
data = data.to(device)

# === Encode all node embeddings ===
with torch.no_grad():
    z_dict = model.encoder(data.x_dict, data.edge_index_dict)

# === Select random investor ===
num_investors = data['investor'].num_nodes
investor_id = random.randint(0, num_investors - 1)
investor_emb = z_dict['investor'][investor_id].unsqueeze(0)  # shape: [1, D]

# === Score all funds ===
fund_embs = z_dict['fund']  # shape: [num_funds, D]
num_funds = fund_embs.size(0)

# Option 1: Dot product scores
scores = torch.matmul(fund_embs, investor_emb.T).squeeze()  # shape: [num_funds]

# Option 2: Use MLP decoder
# scores = model.predictor(investor_emb.expand(num_funds, -1), fund_embs)

# === Top-k recommendations ===
k = 10
topk_indices = scores.topk(k).indices.cpu().numpy()
print(f"\nTop-{k} fund recommendations for investor {investor_id}:\n")
for idx, fund_id in enumerate(topk_indices):
    print(f"{idx+1:>2}. Fund ID: {fund_id}, Score: {scores[fund_id].item():.4f}")
