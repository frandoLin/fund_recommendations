# Fund Recommendations

## Realism in Simulated Financial Dataset

This project simulates realistic financial advisor and fund data to train and evaluate ML models like Sequential Attention Networks and GNNs. The distributions of various features have been crafted to mimic real-world data patterns observed in asset management and financial product recommendation systems.

### What Makes These Distributions Realistic:

#### 1. **Investor AUM (Assets Under Management)**
- Simulated using a **log-normal distribution**.
- Reflects that most advisors manage <$10M, while a small number manage very large portfolios.
- Log-scaled histogram shows long-tail behavior common in wealth distribution.

#### 2. **Investor Engagement Score**
- Sampled from a **Beta(2, 5)** distribution.
- Realistic right-skew: most users are lightly engaged, and a few power users are highly active.
- Matches real-world marketing/funnel dynamics.

#### 3. **Region Distribution**
- Weighted sampling emphasizes **North America** (50%), with diminishing representation in Europe, Asia, Middle East, and South America.
- Mirrors actual global distribution of financial advisors and fund clients.

#### 4. **Fund Strategy Focus**
- Biased toward **Growth** and **Value** funds, which are more common and widely marketed.
- Underrepresented categories like Balanced, Aggressive match real fund catalogs.

#### 5. **Number of Investments per Investor**
- Distribution follows a **Gaussian bell-curve**, centered around ~10.
- Reflects typical investor behavior building a diversified portfolio of 8–15 products.

### Files Generated
- `data/investors.csv` – 3000 advisors with detailed profiles
- `data/funds.csv` – 500 financial products
- `data/investor_product_edges.csv` – ~25,000 historical investor-fund links
- `data/plots/` – Visualizations of feature distributions

---