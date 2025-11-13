# 🚀 Revnet Agent-Based Model Dashboard

Professional, interactive dashboard for exploring Revnet dynamics with agent-based modeling and real-time Plotly visualizations.

![Dashboard Preview](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## ✨ Features

### 🎛️ **Highly Customizable**
- **3 Built-in Archetypes**: Token Launchpad, Stable Commerce, Periodic Fundraising
- **Full Parameter Control**: Adjust all stage parameters, agent distributions, and initial conditions
- **Agent-Based Modeling**: Simulate 6 different agent types with heterogeneous strategies

### 📊 **Professional Visualizations**
- **Interactive Plotly Charts**: Real-time, zoomable, hoverable visualizations
- **Market Dynamics**: Floor price, issuance price, spreads, and trading volumes
- **Agent Analytics**: Portfolio distributions, holdings by type, top performers
- **Transaction History**: Complete audit trail with filtering and export

### 💾 **Export & Analysis**
- Download configuration as JSON
- Export transactions to CSV
- Export time series data for external analysis
- Real-time performance metrics

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the repository:**
```bash
cd /path/to/Revnet
```

2. **Install dependencies:**
```bash
pip install -r requirements_dashboard.txt
```

3. **Run the dashboard:**
```bash
streamlit run streamlit_dashboard.py
```

4. **Open in browser:**
The dashboard will automatically open at `http://localhost:8501`

## 📖 Usage Guide

### 1. **Select an Archetype**
Choose from predefined configurations or build a custom setup:
- **Token Launchpad**: High volatility, speculative launch scenario
- **Stable Commerce**: Low volatility, business loyalty program
- **Periodic Fundraising**: Moderate volatility, timed fundraising rounds
- **Custom**: Full control over all parameters

### 2. **Configure Stage Parameters**
- **Initial Price**: Starting issuance price per token
- **Price Cut**: Controls price escalation (gamma = 1/(1-cut))
- **Price Step**: Days between price adjustments
- **Cash-out Tax**: Redemption tax percentage
- **Split Ratio**: Per-mint revenue distribution
- **Protocol Fee**: Platform fee percentage

### 3. **Set Initial State**
- **Initial Treasury**: Starting base asset reserves
- **Initial Supply**: Starting token supply

### 4. **Configure Agent Population**
- **Number of Agents**: Total simulated agents (10-500)
- **Agent Distribution**: Percentage of each agent type:
  - **Random**: Stochastic buy/sell behavior
  - **Price Sensitive**: Value-based trading
  - **Floor Trader**: Arbitrage on spreads
  - **Momentum**: Trend-following strategy
  - **Hodler**: Long-term accumulation
  - **Arbitrageur**: Profit from mispricings

### 5. **Run Simulation**
- **Duration**: Simulation time horizon (30-730 days)
- **Time Step**: Resolution (0.01-1.0, smaller = more accurate)
- **Random Seed**: For reproducible results

### 6. **Explore Results**
Navigate through 4 comprehensive tabs:
- **📈 Market Dynamics**: Price evolution, trading activity, spreads
- **👥 Agent Activity**: Portfolio distributions, holdings by type
- **💰 Transactions**: Complete transaction history and analysis
- **📊 Analytics**: Volatility metrics, exports, and insights

## 🎯 Example Scenarios

### Scenario 1: High-Volatility Token Launch
```
Archetype: Token Launchpad
Initial Price: $0.10
Price Cut: 20%
Duration: 180 days
Agent Mix: 25% Price Sensitive, 20% Random, 15% each others
```

### Scenario 2: Stable Business Token
```
Archetype: Stable Commerce
Initial Price: $1.00
Price Cut: 0.5%
Duration: 365 days
Agent Mix: 40% Random, 30% Price Sensitive, 30% Hodler
```

### Scenario 3: Periodic Fundraising
```
Archetype: Periodic Fundraising
Initial Price: $1.00
Duration: 360 days
Agent Mix: 30% Price Sensitive, 25% Momentum, balanced others
```

## 🔧 Advanced Features

### Custom Agent Strategies
Each agent type has parameterized behavior:
- **Random**: Poisson arrivals with exponential transaction sizes
- **Price Sensitive**: Trades based on perceived value vs market price
- **Floor Trader**: Exploits spread between issuance and floor
- **Momentum**: Follows recent floor price trends
- **Hodler**: Steady accumulation, rare selling
- **Arbitrageur**: Exploits mispricings with profit thresholds

### Real-Time Metrics
- Treasury growth percentage
- Supply expansion
- Floor price appreciation
- Transaction volume and velocity
- Portfolio value distributions
- Price volatility measures

### Export Capabilities
- **JSON Config**: Complete parameter set for reproducibility
- **CSV Transactions**: Full transaction log with timestamps
- **CSV Time Series**: Aggregate rate functions over time

## 🚢 Deployment Options

### Local Development
```bash
streamlit run streamlit_dashboard.py
```

### Streamlit Cloud (Free Hosting)
1. Push code to GitHub repository
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click
4. Share public URL

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_dashboard.txt .
RUN pip install -r requirements_dashboard.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t revnet-dashboard .
docker run -p 8501:8501 revnet-dashboard
```

### Cloud Platforms
- **Heroku**: Use Streamlit buildpack
- **AWS/GCP/Azure**: Deploy container or use managed app services
- **Render**: Direct Streamlit support

## 📁 Project Structure

```
Revnet/
├── streamlit_dashboard.py          # Main dashboard application
├── requirements_dashboard.txt      # Python dependencies
├── README_DASHBOARD.md            # This file
├── revnet_simulator.py            # Core simulator (required)
├── agent_based_model.py           # ABM implementation (required)
├── analysis.py                    # Analytics tools (optional)
└── visualization.py               # Matplotlib plotting (optional)
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

### Memory Issues with Large Simulations
- Reduce number of agents
- Increase time step (dt)
- Decrease duration

### Slow Performance
- Install `numba` for JIT compilation: `pip install numba`
- Use larger time steps (0.5 or 1.0)
- Reduce agent count for quick iterations

## 🤝 Contributing

Suggestions for improvements:
1. Additional agent strategies
2. Multi-stage simulations
3. Comparative analysis tools
4. Real-time simulation streaming
5. Parameter optimization tools

## 📝 License

Part of the Revnet project by CryptoEconLab.

## 🔗 Resources

- [Revnet Documentation](https://docs.revnet.eth)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Documentation](https://plotly.com/python/)
- [Agent-Based Modeling Guide](https://mesa.readthedocs.io)

## 💡 Tips for Best Results

1. **Start Simple**: Begin with preset archetypes, then customize
2. **Iterate Quickly**: Use larger time steps (0.5-1.0) for fast iteration
3. **Compare Scenarios**: Export configs and re-run with variations
4. **Analyze Agents**: Check portfolio distributions to understand dynamics
5. **Export Data**: Download CSVs for deeper analysis in Jupyter/R

---

**Built with ❤️ using Streamlit and Plotly**

For questions or issues, please open a GitHub issue or contact the CryptoEconLab team.
