# preprocessing/build_graph.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import os

def build_hetero_graph():
    investors = pd.read_csv('../data/investors.csv')
    funds = pd.read_csv('../data/funds.csv')
    edges = pd.read_csv('../data/investor_product_edges.csv')

    data = HeteroData()

    # Define features for investors
    investor_features = investors[['engagement_score', 'past_investment_amount', 'fund_eligibility_count', 'aum']]
    region_enc = pd.get_dummies(investors['region'], prefix='region')
    strategy_enc = pd.get_dummies(investors['investment_strategy'], prefix='strategy')
    investor_features = pd.concat([investor_features, region_enc, strategy_enc], axis=1)
    investor_tensor = investor_features.astype(np.float32).values
    data['investor'].x = torch.tensor(investor_tensor, dtype=torch.float32)

    # Define features for funds
    fund_features = funds[['min_investment']]
    category_enc = pd.get_dummies(funds['category'], prefix='cat')
    strategy_enc = pd.get_dummies(funds['strategy_focus'], prefix='strategy')
    fund_features = pd.concat([fund_features, category_enc, strategy_enc], axis=1)
    fund_tensor = fund_features.astype(np.float32).values
    data['fund'].x = torch.tensor(fund_tensor, dtype=torch.float32)

    # Build edge index
    investor_id_map = {inv_id: i for i, inv_id in enumerate(investors['investor_id'])}
    fund_id_map = {fid: i for i, fid in enumerate(funds['fund_id'])}

    edge_index = [
        [investor_id_map[row['investor_id']] for _, row in edges.iterrows()],
        [fund_id_map[row['fund_id']] for _, row in edges.iterrows()]
    ]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data['investor', 'invests', 'fund'].edge_index = edge_index
    data['fund', 'invested_by', 'investor'].edge_index = edge_index.flip(0)  # Add reverse edge

    print("Heterogeneous graph built:")
    print(data)

    os.makedirs('../output/graph', exist_ok=True)
    torch.save(data, '../output/graph/hetero_graph.pt')
    print("Saved to output/graph/hetero_graph.pt")

if __name__ == '__main__':
    build_hetero_graph()