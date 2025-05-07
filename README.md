# Community-Aware Portfolio Optimization

This repository presents a dynamic, network-based approach to portfolio construction that leverages **community detection on financial correlation networks** . By identifying latent structures in asset co-movements, the strategy aims to improve diversification, reduce turnover, and enhance interpretability in portfolio design.

## Project Summary

* **Graph-based modeling** of stock correlations using rolling windows
* **Minimum Spanning Tree (MST)** filtering to denoise correlation matrices
* **Louvain community detection** to uncover clusters of co-moving assets
* **Event-driven rebalancing** triggered by structural shifts (births, deaths, persistence changes)
* **Portfolio construction** using centrality-weighted representatives and shrinkage-enhanced mean–variance optimization
* **Evaluation** based on cumulative returns, Sharpe ratios, drawdowns, and structural metrics (entropy, persistence)

## Project Structure

```
├── data/                   # Raw and processed financial data
├── results/                  # Generated figures and network diagrams
├── utils/                    # Source code for network construction, analysis, optimization
│   ├── data.py
│   ├── community_tools.py
│   ├── tracker.py
│   └── utils.py
├── main.py
├── community_analysis.ipynb         # Summary of backtest results across configurations
├── results_analysi.ipynb  
├── config.yml
├── requirements.txt        # Python dependencies
└── README.md               # You're here!

```

## Installation

```
git clone https://github.com/SoniaBorsi/Financial-Network-Analysis.git
cd Financial-Network-Analysis
pip install -r requirements.txt

```

Or run individual components for analysis:

```
python3 main.py
```

NOTE: All parameters for data preprocessing, community detection, and portfolio construction are set in a YAML configuration file (`config.yml`). This makes the pipeline easily adjustable without modifying the source code

## Author

- [Sonia Borsi](https://github.com/SoniaBorsi) (Dataism Laboratory of Quantitative Finance, Virginia Tech)
