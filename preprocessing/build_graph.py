import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def build_graph():
    investors = pd.read_csv('../data/investors.csv')
    funds = pd.read_csv('../data/funds.csv')
    edges = pd.read_csv('../data/investor_product_edges.csv')

    # Preprocess investor features
    region_enc = LabelEncoder()
    strategy_enc = LabelEncoder()

    investors['region_enc'] = region_enc.fit_transform(investors['region'])
    investors['investment_strategy_enc'] = strategy_enc.fit_transform(investors['investment_strategy'])

    scaler = StandardScaler()
    investors[['engagement_score', 'past_investment_amount', 'fund_eligibility_count', 'aum']] = scaler.fit_transform(
        investors[['engagement_score', 'past_investment_amount', 'fund_eligibility_count', 'aum']]
    )

    investor_features = torch.tensor(investors[['engagement_score', 'past_investment_amount', 'fund_eligibility_count', 'aum', 'region_enc', 'investment_strategy_enc']].values, dtype=torch.float)

    # Preprocess fund features
    category_enc = LabelEncoder()
    strategy_enc_fund = LabelEncoder()

    funds['category_enc'] = category_enc.fit_transform(funds['category'])
    funds['strategy_focus_enc'] = strategy_enc_fund.fit_transform(funds['strategy_focus'])

    scaler_fund = StandardScaler()
    funds[['min_investment']] = scaler_fund.fit_transform(funds[['min_investment']])

    fund_features = torch.tensor(funds[['min_investment', 'category_enc', 'strategy_focus_enc']].values, dtype=torch.float)

    # Build Heterogeneous Graph
    data = HeteroData()

    data['investor'].x = investor_features
    data['fund'].x = fund_features

    # Map IDs to indices
    investor_id_to_idx = {inv_id: idx for idx, inv_id in enumerate(investors['investor_id'])}
    fund_id_to_idx = {fund_id: idx for idx, fund_id in enumerate(funds['fund_id'])}

    # Edge index
    edge_index_investor_to_fund = torch.tensor(
        [[investor_id_to_idx[row['investor_id']], fund_id_to_idx[row['fund_id']]] for idx, row in edges.iterrows()],
        dtype=torch.long
    ).t().contiguous()

    data['investor', 'invests', 'fund'].edge_index = edge_index_investor_to_fund

    # Save the processed graph
    os.makedirs('../output/graph', exist_ok=True)
    torch.save(data, '../output/graph/hetero_graph.pt')

    print("Heterogeneous Graph saved to output/graph/hetero_graph.pt")

if __name__ == '__main__':
    build_graph()
