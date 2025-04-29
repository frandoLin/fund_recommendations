# train/train_lead_model.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sequential_attention_model import SequentialAttentionDNN


class InvestorDataset(Dataset):
    def __init__(self, df):
        self.x_num = torch.tensor(df[['engagement_score', 'past_investment_amount', 'fund_eligibility_count', 'aum']].values, dtype=torch.float32)
        self.region = torch.tensor(df['region_enc'].values, dtype=torch.long)
        self.strategy = torch.tensor(df['investment_strategy_enc'].values, dtype=torch.long)
        self.y = torch.tensor(df['is_lead'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.region[idx], self.strategy[idx], self.y[idx]

def main():
    df = pd.read_csv('../data/investors.csv')

    # Simulate target: top 25% engagement score → likely lead
    threshold = df['engagement_score'].quantile(0.75)
    df['is_lead'] = (df['engagement_score'] > threshold).astype(int)

    # Encode categorical
    region_enc = LabelEncoder()
    strategy_enc = LabelEncoder()
    df['region_enc'] = region_enc.fit_transform(df['region'])
    df['investment_strategy_enc'] = strategy_enc.fit_transform(df['investment_strategy'])

    # Normalize numeric columns
    scaler = StandardScaler()
    df[['engagement_score', 'past_investment_amount', 'fund_eligibility_count', 'aum']] = scaler.fit_transform(
        df[['engagement_score', 'past_investment_amount', 'fund_eligibility_count', 'aum']])

    # Split into train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    train_ds = InvestorDataset(train_df)
    val_ds = InvestorDataset(val_df)
    test_ds = InvestorDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = SequentialAttentionDNN(region_vocab_size=df['region_enc'].nunique(),
                                   strategy_vocab_size=df['investment_strategy_enc'].nunique())
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x_num, region, strategy, y in train_loader:
            optimizer.zero_grad()
            preds = model(x_num, region, strategy)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loop
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_num, region, strategy, y in val_loader:
                preds = model(x_num, region, strategy)
                val_preds.extend(preds.squeeze().tolist())
                val_labels.extend(y.squeeze().tolist())

        val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds]
        val_acc = accuracy_score(val_labels, val_preds_bin)

        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    # Save model
    os.makedirs('../output/model', exist_ok=True)
    torch.save(model.state_dict(), '../output/model/lead_dnn.pth')

    # Final Test Evaluation
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x_num, region, strategy, y in test_loader:
            preds = model(x_num, region, strategy)
            test_preds.extend(preds.squeeze().tolist())
            test_labels.extend(y.squeeze().tolist())

    test_preds_bin = [1 if p > 0.5 else 0 for p in test_preds]
    roc_auc = roc_auc_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds_bin)
    recall = recall_score(test_labels, test_preds_bin)

    print("\nFinal Test Metrics:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == '__main__':
    main()
