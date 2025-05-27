# train/train_gnn_link_prediction.py
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, precision_recall_curve
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn_model import GNNLinkPredictor
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage

# Add safe globals for serialization
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage])

# Load the graph
data = torch.load('../output/graph/hetero_graph.pt')

# Normalize features (optional but recommended)
for ntype in ['investor', 'fund']:
    x = data[ntype].x
    data[ntype].x = (x - x.mean(0)) / (x.std(0) + 1e-6)

# Extract metadata for hetero GNN
metadata = data.metadata()
print(f"Metadata: {metadata}")

# Positive edge indices (investor-fund pairs)
edge_index = data['investor', 'invests', 'fund'].edge_index
num_pos_edges = edge_index.size(1)

# Generate negative edges
neg_edge_index = negative_sampling(
    edge_index=edge_index,
    num_nodes=(data['investor'].num_nodes, data['fund'].num_nodes),
    num_neg_samples=num_pos_edges
)

# Sanity check: ensure indices are in bounds
assert neg_edge_index[0].max().item() < data['investor'].num_nodes
assert neg_edge_index[1].max().item() < data['fund'].num_nodes

# Combine positive and negative samples
pos_labels = torch.ones(num_pos_edges, dtype=torch.float32)
neg_labels = torch.zeros(num_pos_edges, dtype=torch.float32)
all_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
all_labels = torch.cat([pos_labels, neg_labels], dim=0)

# Shuffle and split
perm = torch.randperm(all_edge_index.size(1))
all_edge_index = all_edge_index[:, perm]
all_labels = all_labels[perm]

n_total = all_edge_index.size(1)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)

train_edge_index = all_edge_index[:, :n_train]
train_labels = all_labels[:n_train]
val_edge_index = all_edge_index[:, n_train:n_train + n_val]
val_labels = all_labels[n_train:n_train + n_val]
test_edge_index = all_edge_index[:, n_train + n_val:]
test_labels = all_labels[n_train + n_val:]

torch.save({
    'test_edge_index': test_edge_index.cpu(),
    'test_labels': test_labels.cpu()
}, '../output/graph/test_edge_indices.pt')
print("Test edges saved to ../output/graph/test_edge_indices.pt for top-k evaluation.")

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNLinkPredictor(hidden_channels=64, metadata=metadata).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_auc = 0
patience = 5
patience_counter = 0

os.makedirs('../output/model', exist_ok=True)

# Training loop
for epoch in range(1, 300):
    model.train()
    optimizer.zero_grad()
    pred = model(data, train_edge_index).view(-1)
    loss = F.binary_cross_entropy(pred, train_labels.to(device))
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(data, val_edge_index).view(-1).cpu()
        val_pred_bin = (val_pred > 0.5).float()
        val_labels_np = val_labels.cpu()

        val_auc = roc_auc_score(val_labels_np, val_pred)
        val_precision = precision_score(val_labels_np, val_pred_bin)
        val_recall = recall_score(val_labels_np, val_pred_bin)

    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), '../output/model/best_gnn_model.pth')
        print("Model improved and saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}!")
            break

# Final test evaluation
model.load_state_dict(torch.load('../output/model/best_gnn_model.pth'))
model.eval()
with torch.no_grad():
    test_pred = model(data, test_edge_index).view(-1).cpu()
    test_pred_bin = (test_pred > 0.5).float()
    test_labels_np = test_labels.cpu()

    test_auc = roc_auc_score(test_labels_np, test_pred)
    test_precision = precision_score(test_labels_np, test_pred_bin)
    test_recall = recall_score(test_labels_np, test_pred_bin)

print("\nFinal Test Metrics:")
print(f"ROC AUC: {test_auc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")

# Save ROC and PR curves
os.makedirs('output/graph', exist_ok=True)

fpr, tpr, _ = roc_curve(test_labels_np, test_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {test_auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.savefig('../output/graph/gnn_roc_curve.png')
plt.show()

precision_curve, recall_curve, _ = precision_recall_curve(test_labels_np, test_pred)
plt.figure()
plt.plot(recall_curve, precision_curve, label=f'PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig('../output/graph/gnn_pr_curve.png')
plt.show()
