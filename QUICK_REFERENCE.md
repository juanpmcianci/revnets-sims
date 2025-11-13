# 🎯 Revnet Dashboard - Quick Reference Card

## 🚀 Launch Commands

```bash
# macOS/Linux
./run_dashboard.sh

# Windows
run_dashboard.bat

# Manual
streamlit run streamlit_dashboard.py
```

---

## 🎛️ Key Parameters Cheat Sheet

| Parameter | Low | Medium | High | Effect |
|-----------|-----|--------|------|--------|
| **Initial Price** | $0.10 | $1.00 | $10.00 | Entry point |
| **Price Cut** | 0-5% | 10-15% | 20-30% | Growth rate |
| **Cash-out Tax** | 2-5% | 10-15% | 20-25% | Exit friction |
| **Split Ratio** | 10-30% | 40-60% | 70-98% | Cashback % |

---

## 👥 Agent Types Summary

| Type | Strategy | Volatility | Holding Period |
|------|----------|------------|----------------|
| **Random** | Noise | Low | Short |
| **Price Sensitive** | Value | Medium | Medium |
| **Floor Trader** | Arbitrage | Medium | Short |
| **Momentum** | Trend | High | Medium |
| **Hodler** | Accumulate | Low | Long |
| **Arbitrageur** | Exploit | High | Very Short |

---

## 📊 Archetype Presets

### Token Launchpad
```
Price: $0.10 → Growth: 25% weekly
Tax: 5% | Split: 15%
Duration: 180 days
Use: Speculative launch
```

### Stable Commerce
```
Price: $1.00 → Growth: 0.5% quarterly
Tax: 2% | Split: 97%
Duration: 365 days
Use: Business loyalty
```

### Periodic Fundraising
```
Price: $1.00 → Growth: 0% (stepped)
Tax: 15% | Split: 20%
Duration: 360 days
Use: Fundraising rounds
```

---

## 🎯 Success Metrics

### Healthy Growth
- ✅ Floor: +2-5% monthly
- ✅ Volatility: < 5%
- ✅ Net Flow: Positive
- ✅ Spread: > 10%

### Stable Operation
- ✅ Floor: ±1% variance
- ✅ Volatility: < 0.5%
- ✅ Flow: Balanced
- ✅ Redemption Rate: High

### Warning Signs
- ⚠️ Declining floor
- ⚠️ Volatility > 15%
- ⚠️ Negative flow
- ⚠️ Spread < 5%

---

## ⚡ Performance Tips

### Fast Testing
```
Agents: 50
Time Step: 0.5-1.0
Duration: 90 days
```

### Production Run
```
Agents: 100-200
Time Step: 0.1
Duration: 365 days
Install: pip install numba
```

---

## 🐛 Quick Fixes

**Port busy?**
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

**Slow simulation?**
- Reduce agents to 50
- Increase dt to 0.5
- Install numba

**Memory error?**
- Reduce agents
- Shorter duration
- Larger time step

---

## 📥 Export Options

1. **Config (JSON)**: Full parameter set
2. **Transactions (CSV)**: Complete audit trail
3. **Time Series (CSV)**: Rate functions

---

## 🔑 Key Formulas

```
gamma = 1 / (1 - price_cut)
floor_price = (1 - protocol_fee) * (1 - cashout_tax) * (treasury / supply)
spread = issuance_price - floor_price
backing_ratio = treasury / supply
```

---

## 📈 Interpretation Guide

**Floor Price Chart**
- Rising = Growth
- Flat = Stability
- Falling = Concern

**Spread Chart**
- Large = Strong demand
- Medium = Healthy
- Small = Ceiling pressure

**Transaction Volume**
- High Cash-In = Demand
- High Cash-Out = Selling
- Balanced = Equilibrium

---

## 💡 Pro Tips

1. **Start with presets** → understand behavior
2. **Test fast** (dt=1.0) → refine slowly (dt=0.1)
3. **Export everything** → external analysis
4. **One change at a time** → isolate effects
5. **Compare runs** → use same seed

---

## 🔗 URLs & Commands

**Dashboard**: http://localhost:8501
**Docs**: README_DASHBOARD.md
**Guide**: DASHBOARD_GUIDE.md

**Install**: `pip install -r requirements_dashboard.txt`
**Update**: `pip install --upgrade streamlit plotly`
**Check**: `streamlit --version`

---

## 🎨 Color Coding

- 🟢 Green = Cash-In / Growth
- 🔴 Red = Cash-Out / Redemption
- 🟣 Purple = Price Sensitive
- 🟠 Orange = Activity Rate
- 🔵 Blue = Treasury
- ⚫ Black = Issuance Price

---

**Print this card for quick reference! 📋**
