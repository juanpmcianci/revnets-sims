# Revnet Dynamical System Simulator - Getting Started

Welcome! This is a complete Python simulation environment for Web3 Revnet mechanisms.

## 📁 Files in This Package

### Core Code (Python Modules)
- **revnet_simulator.py** - Main ODE solver and stage management
- **agent_based_model.py** - Agent-based modeling framework
- **analysis.py** - Analytical solutions, counterfactuals, UQ, sensitivity analysis
- **visualization.py** - Comprehensive plotting functions

### Getting Started
- **test_installation.py** - Run this first to verify everything works
- **quickstart.py** - Interactive tutorial (start here!)
- **examples.py** - 7 comprehensive examples demonstrating all features

### Documentation
- **README.md** - Complete API documentation and user guide
- **PROJECT_SUMMARY.md** - Technical overview and design decisions
- **requirements.txt** - Python dependencies
- **INDEX.md** - This file

### Example Outputs (PNG files)
- example1_basic_simulation.png
- example2_analytical_comparison.png
- example3_counterfactual.png
- example4_parameter_sweep.png
- example5_uncertainty.png
- example6_sensitivity.png
- example7_agent_based.png
- quickstart_simulation.png

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy scipy matplotlib seaborn
```

### Step 2: Verify Installation
```bash
python test_installation.py
```

You should see "ALL TESTS PASSED ✓"

### Step 3: Run the Tutorial
```bash
python quickstart.py
```

This will walk you through your first simulation and explain key concepts.

## 📚 What to Read Next

### For Beginners
1. Start with **quickstart.py** (run it and read the code)
2. Read the "Quick Start" section in **README.md**
3. Try modifying parameters in quickstart.py

### For Advanced Users
1. Read **PROJECT_SUMMARY.md** for technical details
2. Run **examples.py** to see all features
3. Read **README.md** API documentation
4. Start building your own simulations

## 🎯 Main Features

### What You Can Do

1. **Simulate Revnet Dynamics**
   - Continuous-time ODE integration
   - Multi-stage configurations
   - Custom rate functions
   - Auto-issuance events

2. **Analyze Scenarios**
   - Compare counterfactuals
   - Sweep parameters
   - Find steady states
   - Compute sensitivities

3. **Quantify Uncertainty**
   - Monte Carlo simulations
   - Latin Hypercube Sampling
   - Confidence intervals
   - Variance decomposition

4. **Model Agent Behavior**
   - 6 agent strategies
   - Heterogeneous populations
   - Emergent dynamics
   - Transaction tracking

5. **Visualize Everything**
   - Time series plots
   - Phase diagrams
   - Uncertainty bands
   - Sensitivity charts

## 💡 Usage Examples

### Minimal Example
```python
from revnet_simulator import Stage, RevnetState, RevnetSimulator, constant_rate

# Define system
stage = Stage(t_start=0.0, P_issue_0=1.0, gamma_cut=0.05, 
              Delta_t=10.0, r_cashout=0.1)
state = RevnetState(B=1000.0, S=1000.0, t=0.0)
sim = RevnetSimulator([stage], state, phi_tot=0.02)

# Run
sim.simulate(100.0, constant_rate(50.0), constant_rate(40.0))

# Results
history = sim.get_history()
print(f"Final floor: {history['P_floor'][-1]:.4f}")
```

### Counterfactual Analysis
```python
from analysis import CounterfactualAnalyzer

analyzer = CounterfactualAnalyzer(stages, initial_state, phi_tot)
results = analyzer.compare_rate_scenarios(t_end=100.0, rate_scenarios={
    "High Growth": (constant_rate(80.0), constant_rate(30.0)),
    "Balanced": (constant_rate(50.0), constant_rate(50.0)),
})
```

### Agent-Based Model
```python
from agent_based_model import AgentBasedRevnetSimulator, create_agent_population, AgentType

agents = create_agent_population(100, {
    AgentType.PRICE_SENSITIVE: 0.4,
    AgentType.HODLER: 0.3,
    AgentType.ARBITRAGEUR: 0.3,
})

abm_sim = AgentBasedRevnetSimulator(stages, initial_state, agents, phi_tot)
abm_sim.simulate(t_end=100.0, dt=0.5)
```

## 🧪 Testing Your Setup

If you encounter issues:

1. **Check Python version**: Requires Python 3.8+
   ```bash
   python --version
   ```

2. **Verify all files are present**:
   ```bash
   ls *.py
   ```
   Should show: revnet_simulator.py, agent_based_model.py, analysis.py, 
                visualization.py, examples.py, quickstart.py, test_installation.py

3. **Run tests**:
   ```bash
   python test_installation.py
   ```

4. **Check dependencies**:
   ```bash
   python -c "import numpy, scipy, matplotlib, seaborn; print('OK')"
   ```

## 📖 Documentation Structure

- **INDEX.md** (this file) - Start here
- **README.md** - Complete reference
- **PROJECT_SUMMARY.md** - Technical details
- **quickstart.py** - Interactive tutorial
- **examples.py** - Feature demonstrations

## 🎓 Learning Path

### Beginner Path
1. Run `test_installation.py`
2. Run and read `quickstart.py`
3. Try modifying parameters in quickstart
4. Read README "Quick Start" section

### Intermediate Path
1. Run `examples.py`
2. Study individual examples
3. Read README "Features" section
4. Try counterfactual analysis

### Advanced Path
1. Read PROJECT_SUMMARY.md
2. Study analysis.py source code
3. Implement custom agent strategies
4. Build your own applications

## 🤔 Common Questions

**Q: What's the difference between continuous-time and agent-based simulation?**
A: Continuous-time uses aggregate rate functions (smooth ODEs). Agent-based simulates individual agents (discrete events). Both are included!

**Q: How accurate is the numerical integration?**
A: < 0.01% error vs analytical solutions. See example2_analytical_comparison.png

**Q: Can I use real-world data?**
A: Yes! Define custom rate functions that interpolate your data.

**Q: How do I add a new agent type?**
A: Extend the Agent class and add logic in compute_agent_action(). See agent_based_model.py

**Q: What if my simulation diverges or crashes?**
A: Check that rates don't cause S → 0. Use smaller dt. Check parameter bounds.

## 📊 Understanding the Outputs

The simulator tracks:
- **B(t)** - Treasury balance (base asset units)
- **S(t)** - Token supply (token units)
- **P_floor(t)** - Marginal redemption floor price
- **P_issue(t)** - Current issuance price

Key relationship:
```
P_floor = (1 - phi_tot) × (1 - r_cashout) × (B / S)
```

At steady state:
```
P_floor = (1 - phi_tot) × P_issue  (independent of r_cashout!)
```

## 🛠️ Troubleshooting

### Import Errors
- Ensure all .py files are in the same directory
- Check Python path: `import sys; print(sys.path)`

### Numerical Issues
- Use smaller dt (e.g., 0.01 instead of 0.1)
- Check for extreme parameter values
- Monitor for S → 0 warnings

### Visualization Issues
- Ensure matplotlib backend is configured
- Use `plt.ioff()` for non-interactive plotting
- Save figures instead of displaying: `plt.savefig('output.png')`

## 🎉 You're Ready!

Start with:
```bash
python quickstart.py
```

Then explore:
```bash
python examples.py
```

Happy simulating! 🚀

---

## 📞 Support

For questions about:
- **Math/theory**: See the original document "Revnet Value Flows as a Continuous-Time Dynamical System"
- **Implementation**: Read README.md and PROJECT_SUMMARY.md
- **Examples**: Study examples.py with comments
- **Bugs**: Check test_installation.py passes

## 📝 Citation

If using this simulator in research:
```
The CEL Team. "Revnet Value Flows as a Continuous-Time Dynamical System:
A Rate-Based Formalization." 2025.
```

## 📄 License

MIT License - See source files for details.

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Status**: Production Ready ✓
