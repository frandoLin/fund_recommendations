import pandas as pd
import numpy as np
import os
import random

def generate_investors(num_investors=3000):
    regions = ['North America', 'Europe', 'Asia', 'Middle East', 'South America']
    region_weights = [0.5, 0.2, 0.15, 0.1, 0.05]  # More North America
    strategies = ['Growth', 'Value', 'Balanced', 'Income', 'Aggressive']
    fund_categories = ['Mutual Funds', 'Hedge Funds', 'Private Equity', 'Fixed Income', 'ETF']

    investors = []
    for i in range(num_investors):
        investor = {
            'investor_id': f'I{i:05d}',
            'engagement_score': round(np.random.beta(2, 5), 2),  # right-skewed
            'region': random.choices(regions, weights=region_weights, k=1)[0],
            'past_investment_amount': round(np.random.uniform(10000, 5000000), 2),
            'fund_eligibility_count': random.randint(1, 5),
            'investment_strategy': random.choice(strategies),
            'aum': round(np.random.lognormal(mean=14.5, sigma=1.0), 2)  # log-normalized AUM
        }
        investors.append(investor)
    return pd.DataFrame(investors)

def generate_funds(num_funds=500):
    categories = ['Mutual Funds', 'Hedge Funds', 'Private Equity', 'Fixed Income', 'ETF']
    strategies = ['Growth'] * 3 + ['Value'] * 3 + ['Income', 'Aggressive', 'Balanced']  # More Growth/Value

    funds = []
    for i in range(num_funds):
        fund = {
            'fund_id': f'F{i:05d}',
            'category': random.choice(categories),
            'strategy_focus': random.choice(strategies),
            'min_investment': round(np.random.uniform(5000, 50000), 2)
        }
        funds.append(fund)
    return pd.DataFrame(funds)

def generate_investor_product_edges(investors, funds, link_probability=0.02):
    edges = []
    for inv_id in investors['investor_id']:
        for fund_id in funds['fund_id']:
            if random.random() < link_probability:
                edges.append({'investor_id': inv_id, 'fund_id': fund_id})
    return pd.DataFrame(edges)

def main():
    os.makedirs('../data', exist_ok=True)

    investors = generate_investors()
    funds = generate_funds()
    edges = generate_investor_product_edges(investors, funds)

    investors.to_csv('../data/investors.csv', index=False)
    funds.to_csv('../data/funds.csv', index=False)
    edges.to_csv('../data/investor_product_edges.csv', index=False)

    print(f"Data generated! {len(investors)} investors, {len(funds)} funds, {len(edges)} investment edges.")

if __name__ == '__main__':
    main()
