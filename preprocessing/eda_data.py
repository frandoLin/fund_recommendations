# evaluate/eda_investor_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Data
investors = pd.read_csv('../data/investors.csv')
funds = pd.read_csv('../data/funds.csv')
edges = pd.read_csv('../data/investor_product_edges.csv')

os.makedirs('../output/plots', exist_ok=True)

# 1. Investor engagement score distribution
plt.figure(figsize=(8,5))
sns.histplot(investors['engagement_score'], bins=30, kde=True)
plt.title('Investor Engagement Score Distribution')
plt.xlabel('Engagement Score')
plt.ylabel('Count')
plt.savefig('../output/plots/investor_engagement_score.png')
plt.close()

# 2. AUM (Assets Under Management) distribution
plt.figure(figsize=(8,5))
sns.histplot(investors['aum'], bins=30, kde=True)
plt.title('Investor AUM Distribution')
plt.xlabel('AUM ($)')
plt.ylabel('Count')
plt.xscale('log')
plt.savefig('../output/plots/investor_aum_distribution.png')
plt.close()

# 3. Investor region distribution
plt.figure(figsize=(8,5))
sns.countplot(data=investors, x='region', order=investors['region'].value_counts().index)
plt.title('Investor Region Counts')
plt.ylabel('Number of Investors')
plt.savefig('../output/plots/investor_region_counts.png')
plt.close()

# 4. Fund strategy focus distribution
plt.figure(figsize=(8,5))
sns.countplot(data=funds, x='strategy_focus', order=funds['strategy_focus'].value_counts().index)
plt.title('Fund Strategy Focus Counts')
plt.ylabel('Number of Funds')
plt.savefig('../output/plots/fund_strategy_focus.png')
plt.close()

# 5. Number of investments per investor
investor_edge_counts = edges['investor_id'].value_counts()
plt.figure(figsize=(8,5))
sns.histplot(investor_edge_counts, bins=30, kde=True)
plt.title('Number of Investments per Investor')
plt.xlabel('Number of Investments')
plt.ylabel('Number of Investors')
plt.savefig('../output/plots/investments_per_investor.png')
plt.close()

# 6. Fund eligibility count
plt.figure(figsize=(8,5))
sns.countplot(x='fund_eligibility_count', data=investors)
plt.title('Fund Eligibility Count per Investor')
plt.xlabel('Number of Eligible Fund Categories')
plt.ylabel('Number of Investors')
plt.savefig('../output/plots/fund_eligibility_count.png')
plt.close()

print("EDA plots saved in ouput/plots/")
