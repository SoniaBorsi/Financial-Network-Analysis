# Network Analysis and Community Detection for Portfolio Optimization

This repository presents a dynamic, network-based approach to portfolio construction that leverages **community detection on financial correlation networks** . By identifying latent structures in asset co-movements, the strategy aims to improve diversification, reduce turnover, and enhance interpretability in portfolio design.
<br>

<p align="center">
  <img src="https://github.com/SoniaBorsi/Financial-Network-Analysis/blob/4e054d904d2a31dfe6ea590abdae4736f686c746/results/MST.png?raw=true" width="512"/>  
</p>
<p align="center">
  <sub><em>Minimum Spanning Tree of SP500 stocks. </em></sub>
</p>


## Table of Contents:

- Project Summary
- Project Structure
- Installation
- Notebooks

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
├── results/                  # Figures and network plots
├── utils/                    # Source code for network construction, analysis, optimization
│   ├── data.py
│   ├── community_tools.py
│   ├── tracker.py
│   └── utils.py
├── main.py
├── community_analysis.ipynb    # Analyzes community evolution and structure
├── results_analysis.ipynb      # Visualizes and evaluates portfolio performance
├── config.yml
├── requirements.txt        # Python dependencies
└── README.md   

```

## Installation

```
git clone https://github.com/SoniaBorsi/Financial-Network-Analysis.git
cd Financial-Network-Analysis
pip install -r requirements.txt

```

Run the main script:

```
python3 main.py
```

NOTE: All parameters for data preprocessing, community detection, and portfolio construction are set in a YAML configuration file (`config.yml`). This makes the pipeline easily adjustable without modifying the source code

## Notebooks

* `community_analysis.ipynb`:
  Visualizes and quantifies the evolution of network communities over time. Supports the *Network Analysis* (Part 1 methodology of the [report]).
* `results_analysis.ipynb`:
* Evaluates the performance of the community-aware portfolio using cumulative returns, Sharpe ratios, volatility, and drawdowns. Supports the *portfolio evaluation* *strategy* and *evaluation (*Part 2 methodology section of the [report]).

## Author

- [Sonia Borsi](https://github.com/SoniaBorsi) (Dataism Laboratory of Quantitative Finance, Virginia Tech)
