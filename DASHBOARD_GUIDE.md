# 📚 Revnet Dashboard - Complete User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding the Interface](#understanding-the-interface)
3. [Configuration Options](#configuration-options)
4. [Interpreting Results](#interpreting-results)
5. [Advanced Tips](#advanced-tips)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Quick Launch (macOS/Linux)
```bash
./run_dashboard.sh
```

### Quick Launch (Windows)
```cmd
run_dashboard.bat
```

### Manual Launch
```bash
pip install -r requirements_dashboard.txt
streamlit run streamlit_dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## Understanding the Interface

### Sidebar: Configuration Panel

The left sidebar contains all simulation controls organized in 5 sections:

#### 1️⃣ **Archetype Selection**
Pre-configured scenarios representing different Revnet use cases:

| Archetype | Use Case | Characteristics |
|-----------|----------|----------------|
| **Token Launchpad** | Speculative launch | High volatility, rapid growth, short duration |
| **Stable Commerce** | Business loyalty | Stable price, high cashback, long duration |
| **Periodic Fundraising** | Timed rounds | Moderate volatility, scheduled increases |
| **Custom** | Full control | Define all parameters manually |

**💡 Tip**: Start with a preset to understand behavior, then customize.

#### 2️⃣ **Stage Parameters**

**Initial Price ($)**: Starting issuance price per token
- Low (< $0.50): Accessible entry, rapid early growth
- Medium ($0.50-$2.00): Balanced approach
- High (> $2.00): Premium positioning

**Price Cut (%)**: Controls price escalation
- Formula: `gamma = 1 / (1 - cut)`
- 0%: No price changes (stable)
- 5-10%: Modest appreciation (~5-11% increase per step)
- 20-30%: Aggressive growth (~25-43% increase per step)

**Price Step (days)**: Time between price adjustments
- Weekly (7): Frequent adjustments, dynamic pricing
- Monthly (30): Moderate cadence
- Quarterly (90): Long-term stability

**Cash-out Tax (%)**: Redemption penalty
- Low (2-5%): Liquid, easily redeemable
- Medium (10-15%): Balanced incentives
- High (20-25%): Strong holding incentive

**Split Ratio (%)**: Revenue distribution
- Low (10-30%): Most to operators/treasury
- Medium (40-60%): Balanced split
- High (70-98%): Loyalty/cashback model

**Protocol Fee (%)**: Platform fee (typically 0-5%)

#### 3️⃣ **Initial State**

**Initial Treasury**: Starting reserves in base asset
**Initial Supply**: Starting token circulation

The ratio `Treasury/Supply` determines the initial floor price.

#### 4️⃣ **Agent Population**

**Number of Agents**: More agents = smoother dynamics but slower simulation
- 10-50: Fast, suitable for testing
- 50-100: Balanced (recommended)
- 100-500: Detailed, realistic behavior

**Agent Type Distribution**:

| Agent Type | Behavior | Best For |
|------------|----------|----------|
| **Random** | Stochastic buy/sell | Market noise, casual users |
| **Price Sensitive** | Value-based trading | Rational investors |
| **Floor Trader** | Arbitrage spreads | Market makers |
| **Momentum** | Trend following | Speculative traders |
| **Hodler** | Long-term accumulation | Core community |
| **Arbitrageur** | Exploit mispricings | Sophisticated traders |

**💡 Tip**: The last agent type automatically adjusts to ensure 100% distribution.

#### 5️⃣ **Simulation Settings**

**Duration (days)**: Simulation time horizon
- Short (30-90): Quick scenarios
- Medium (180-365): Standard analysis
- Long (365-730): Long-term projections

**Time Step**: Resolution vs speed trade-off
- 0.01-0.05: High accuracy, slower
- 0.1: Balanced (recommended)
- 0.5-1.0: Fast, good for testing

**Random Seed**: For reproducible results

---

## Configuration Options

### Main Display Area

After running a simulation, results appear in 4 tabs:

### Tab 1: 📈 Market Dynamics

**What you see:**
1. **Floor vs Issuance Price**: Shows the price corridor
   - Red dashed line: Issuance ceiling
   - Green solid line: Redemption floor
   - Gap = opportunity for trading

2. **Cash-In Activity**: Demand for tokens
   - Higher = more buying pressure
   - Varies by agent behavior

3. **Cash-Out Activity**: Redemption requests
   - Higher = selling pressure
   - Reduced by cash-out tax

4. **Price Spread**: Premium over floor
   - Large spread = strong demand
   - Small spread = approaching ceiling

**How to interpret:**
- Rising floor + large spread = healthy growth
- Stable floor + small spread = equilibrium
- Falling floor = excessive redemptions

### Tab 2: 👥 Agent Activity

**What you see:**
1. **Token Holdings by Agent Type**: Who holds what
2. **Portfolio Value Distribution**: Wealth concentration
3. **Top 10 Agents**: Biggest winners

**How to interpret:**
- Concentrated holdings = whale risk
- Diverse holdings = healthy distribution
- Agent type patterns reveal strategy effectiveness

### Tab 3: 💰 Transactions

**What you see:**
1. **Transaction Timeline**: Visual history
   - Green dots = Cash-ins
   - Red dots = Redemptions

2. **Volume Statistics**: Aggregate flows

3. **Recent Transactions**: Audit trail

**How to interpret:**
- Clustered transactions = coordinated behavior
- Steady flow = organic activity
- Net positive flow = growth phase

### Tab 4: 📊 Analytics

**What you see:**
1. **Volatility Metrics**: Price stability measures
2. **Performance Summary**: Key ratios
3. **Export Options**: Download data

**Advanced metrics:**
- **Volatility < 1%**: Very stable
- **Volatility 1-5%**: Moderate
- **Volatility > 10%**: High volatility
- **Backing Ratio**: Safety margin

---

## Interpreting Results

### Success Indicators

✅ **Healthy Growth**
- Floor price steadily increasing
- Moderate volatility (< 5%)
- Positive net flow
- Diverse agent holdings
- Spread stays above 10%

✅ **Stable Operation**
- Floor near $1.00 ± 2%
- Very low volatility (< 0.5%)
- Balanced cash-in/out
- High redemption rate (low friction)

✅ **Successful Launch**
- Rapid floor appreciation (> 50%)
- High initial cash-in
- Declining redemptions
- Momentum trader accumulation

### Warning Signs

⚠️ **Potential Issues**
- Floor declining over time
- Extreme volatility (> 15%)
- Negative net flow sustained
- Single agent type dominance
- Spread approaching zero

---

## Advanced Tips

### Experiment Design

**Comparing Scenarios**:
1. Run baseline simulation
2. Export config (JSON)
3. Change ONE parameter
4. Re-run and compare
5. Export time series for plotting

**Parameter Sensitivity**:
- **Cash-out tax**: Test 5%, 10%, 15%, 20%
- **Agent mix**: Pure vs diverse populations
- **Price steps**: Weekly vs monthly impacts

### Optimal Configurations

**For Stability**:
```
Price Cut: 0-1%
Cash-out Tax: 2-5%
Split Ratio: 90-98%
Agents: 60% Random, 40% Hodler
```

**For Growth**:
```
Price Cut: 10-20%
Cash-out Tax: 10-20%
Split Ratio: 20-40%
Agents: Balanced mix with Momentum
```

**For Liquidity**:
```
Price Cut: 5-10%
Cash-out Tax: 2-8%
Split Ratio: 50%
Agents: High Arbitrageur (20%+)
```

### Performance Optimization

**Faster Simulations**:
- Use dt = 0.5 or 1.0
- Reduce agents to 50
- Shorter duration (90 days)

**More Realistic**:
- Use dt = 0.1 or 0.05
- 100+ agents
- Longer duration (365+ days)
- Install `numba` for 2-3x speedup

---

## Troubleshooting

### Common Issues

**"No module named 'streamlit'"**
```bash
pip install -r requirements_dashboard.txt
```

**"Port 8501 already in use"**
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

**Simulation runs forever**
- Reduce agents or increase time step
- Check for parameter errors (e.g., gamma_cut >= 1.0)

**Charts not displaying**
- Ensure plotly is installed: `pip install plotly`
- Clear browser cache
- Try different browser

**Memory errors**
- Reduce agents to 50 or less
- Increase time step to 0.5 or 1.0
- Reduce duration to 180 days or less

### Getting Help

1. Check README_DASHBOARD.md for setup
2. Review this guide for usage
3. Open GitHub issue with:
   - Error message
   - Configuration (export JSON)
   - Python version: `python --version`
   - Package versions: `pip list | grep -E "streamlit|plotly|numpy"`

---

## Keyboard Shortcuts

- **R**: Rerun simulation
- **Ctrl+Shift+R**: Clear cache and rerun
- **Ctrl+S**: Open settings
- **?**: Show keyboard shortcuts

---

## Best Practices

1. **Start Simple**: Use presets before customizing
2. **Iterate**: Run quick tests with dt=1.0, then refine
3. **Compare**: Export configs and run A/B tests
4. **Document**: Download CSVs for your records
5. **Visualize**: Use exported data in Jupyter for deeper analysis

---

## Example Workflows

### Workflow 1: Testing New Archetype
1. Select "Custom" archetype
2. Set desired parameters
3. Run with 50 agents, dt=0.5, 90 days
4. Review results
5. Adjust parameters
6. Re-run for comparison
7. Once satisfied, run full simulation (100 agents, dt=0.1, 365 days)

### Workflow 2: Agent Strategy Analysis
1. Start with "Token Launchpad" preset
2. Run baseline (balanced agent mix)
3. Export config and results
4. Change to 100% single agent type
5. Compare performance across agent types
6. Identify best-performing strategies

### Workflow 3: Sensitivity Analysis
1. Pick stable baseline (e.g., "Stable Commerce")
2. Run and record floor price at t=365
3. Vary ONE parameter (e.g., cash-out tax: 2%, 5%, 10%, 15%, 20%)
4. Record final floor for each
5. Plot sensitivity curve externally
6. Identify optimal parameter value

---

## Additional Resources

- **Revnet Docs**: Technical specifications
- **ABM Theory**: Mesa framework documentation
- **Plotly Docs**: Customizing visualizations
- **Streamlit Docs**: Dashboard customization

---

**Happy Experimenting! 🚀**

For advanced customization of agent strategies, see `agent_based_model.py` source code.
