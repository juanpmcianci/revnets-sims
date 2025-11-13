# Revnet Dynamical System Simulator

A comprehensive Python simulation environment for Web3 Revnet mechanisms with continuous-time dynamics, agent-based modeling, uncertainty quantification, and counterfactual analysis.

## Overview

This simulator implements the mathematical framework described in "Revnet Value Flows as a Continuous-Time Dynamical System: A Rate-Based Formalization". It provides:

- **Continuous-time ODE solver** for Revnet dynamics
- **Agent-based modeling** with heterogeneous agent strategies
- **Analytical solutions** for validation
- **Counterfactual analysis** for comparing scenarios
- **Uncertainty quantification** via Monte Carlo and Latin Hypercube Sampling
- **Sensitivity analysis** using variance-based methods
- **Parameter sweeps** for design exploration
- **Comprehensive visualization** tools

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib seaborn --break-system-packages
```

### Files

- `revnet_simulator.py` - Core ODE simulation engine
- `agent_based_model.py` - Agent-based modeling framework
- `analysis.py` - Analysis and UQ tools
- `visualization.py` - Plotting functions
- `examples.py` - Comprehensive examples

## Quick Start

```python
from revnet_simulator import Stage, RevnetState, RevnetSimulator, constant_rate
from visualization import plot_simulation_results

# Define a stage
stage = Stage(
    t_start=0.0,
    P_issue_0=1.0,      # Initial issuance price
    gamma_cut=0.05,     # 5% price cut per step
    Delta_t=10.0,       # Price step interval
    r_cashout=0.1,      # 10% cash-out tax
)

# Initial state
initial_state = RevnetState(B=1000.0, S=1000.0, t=0.0)

# Create simulator
sim = RevnetSimulator(
    stages=[stage],
    initial_state=initial_state,
    phi_tot=0.02,  # 2% protocol fee
    seed=42
)

# Run simulation
sim.simulate(
    t_end=100.0,
    r_in_func=constant_rate(50.0),   # Cash-in rate
    r_out_func=constant_rate(40.0),  # Cash-out rate
    dt=0.1
)

# Visualize
history = sim.get_history()
plot_simulation_results(history)
```

## Core Concepts

### Stages

A `Stage` represents a period with fixed parameters:

```python
Stage(
    t_start=0.0,                          # Start time
    P_issue_0=1.0,                        # Initial issuance price
    gamma_cut=0.05,                       # Price cut factor
    Delta_t=10.0,                         # Time between price steps
    sigma=0.1,                            # Per-mint split
    r_cashout=0.1,                        # Cash-out tax
    auto_issuances=[(50.0, 100.0)]       # (time, amount) pairs
)
```

The issuance price evolves as:
```
P_issue(t) = P_issue_0 * gamma^floor((t - t_start) / Delta_t)
where gamma = 1 / (1 - gamma_cut)
```

### Dynamical System

The system state `(B, S)` evolves according to:

```
dS/dt = r_in(t) / P_issue(t) - r_out(t)
dB/dt = r_in(t) - (1 - r_cashout) * (B/S) * r_out(t)
```

The marginal redemption floor is:
```
P_floor(t) = (1 - phi_tot) * (1 - r_cashout) * B(t) / S(t)
```

### Rate Functions

Rate functions can be:
- Constant: `constant_rate(value)`
- Piecewise: `piecewise_rate(time_points, rates)`
- Sinusoidal: `sinusoidal_rate(base, amplitude, period, phase)`
- Custom: any `Callable[[float], float]`

## Features

### 1. Basic Simulation

```python
from revnet_simulator import RevnetSimulator, constant_rate

sim = RevnetSimulator(stages, initial_state, phi_tot)
sim.simulate(t_end=100.0, r_in_func, r_out_func, dt=0.1)
history = sim.get_history()
```

### 2. Analytical Solutions

```python
from analysis import AnalyticalSolver

# Get closed-form solution
S_t, B_t = AnalyticalSolver.solve_constant_rates(
    B_0, S_0, r_in, r_out, P_issue, r_cashout, t
)

# Compute floor derivative
dP_floor = AnalyticalSolver.floor_derivative(
    B, S, r_in, r_out, P_issue, r_cashout
)
```

### 3. Agent-Based Modeling

```python
from agent_based_model import (
    AgentBasedRevnetSimulator, create_agent_population, AgentType
)

# Create heterogeneous agents
agents = create_agent_population(
    n_agents=100,
    agent_type_distribution={
        AgentType.PRICE_SENSITIVE: 0.4,
        AgentType.HODLER: 0.3,
        AgentType.ARBITRAGEUR: 0.3,
    },
    seed=42
)

# Run ABM
abm_sim = AgentBasedRevnetSimulator(stages, initial_state, agents, phi_tot)
abm_sim.simulate(t_end=100.0, dt=0.5)
```

**Agent Types:**
- `RANDOM` - Random buy/sell with Poisson arrivals
- `PRICE_SENSITIVE` - Buy when price is low, sell when high
- `FLOOR_TRADER` - Trade based on issuance-floor spread
- `MOMENTUM` - Follow price trends
- `HODLER` - Long-term holder, rarely sells
- `ARBITRAGEUR` - Exploit mispricings

### 4. Counterfactual Analysis

```python
from analysis import CounterfactualAnalyzer

analyzer = CounterfactualAnalyzer(stages, initial_state, phi_tot)

# Compare scenarios
scenarios = {
    "High Issuance": (constant_rate(100.0), constant_rate(30.0)),
    "Balanced": (constant_rate(50.0), constant_rate(50.0)),
    "High Redemption": (constant_rate(30.0), constant_rate(60.0)),
}

results = analyzer.compare_rate_scenarios(t_end=100.0, rate_scenarios=scenarios)
```

### 5. Parameter Sweeps

```python
# Sweep a parameter
results = analyzer.parameter_sweep(
    t_end=100.0,
    param_name='r_cashout',
    param_values=np.linspace(0.0, 0.5, 11),
    r_in_func=constant_rate(50.0),
    r_out_func=constant_rate(40.0)
)
```

### 6. Uncertainty Quantification

```python
from analysis import UncertaintyQuantifier

quantifier = UncertaintyQuantifier(stages, initial_state, phi_tot)

# Monte Carlo with uncertain rates
def r_in_sampler(rng):
    rate = rng.normal(50.0, 10.0)
    return constant_rate(max(0, rate))

uq_results = quantifier.monte_carlo_rates(
    t_end=100.0,
    r_in_sampler=r_in_sampler,
    r_out_sampler=r_out_sampler,
    n_samples=100
)

# Access statistics
print(f"Mean floor: {uq_results['P_floor_mean'][-1]:.4f}")
print(f"Std floor: {uq_results['P_floor_std'][-1]:.4f}")
print(f"90% CI: [{uq_results['P_floor_q05'][-1]:.4f}, {uq_results['P_floor_q95'][-1]:.4f}]")
```

### 7. Sensitivity Analysis

```python
from analysis import SensitivityAnalyzer

# Latin Hypercube Sampling
lhs_results = quantifier.latin_hypercube_sampling(
    t_end=100.0,
    param_distributions={
        'r_cashout': (0.0, 0.3),
        'gamma_cut': (0.01, 0.10),
        'r_in': (30.0, 70.0),
        'r_out': (20.0, 60.0),
    },
    base_r_in=50.0,
    base_r_out=40.0,
    n_samples=200
)

# Compute Sobol indices
sobol_indices = SensitivityAnalyzer.compute_sobol_indices(lhs_results)
```

### 8. Visualization

```python
from visualization import (
    plot_simulation_results,
    plot_counterfactual_comparison,
    plot_uncertainty_quantification,
    plot_parameter_sweep,
    plot_sensitivity_analysis,
    plot_agent_behavior
)

# Basic simulation plot
plot_simulation_results(history, title="My Simulation")

# Counterfactual comparison
plot_counterfactual_comparison(results)

# Uncertainty bands
plot_uncertainty_quantification(uq_results)

# Parameter sweep
plot_parameter_sweep(sweep_results, param_name='r_cashout')

# Sensitivity bars
plot_sensitivity_analysis(sobol_indices)

# Agent behavior
plot_agent_behavior(abm_simulator)
```

## Mathematical Background

### Steady State

At steady state with constant rates and issuance price:

```
r_out* = r_in / P_issue
B*/S* = P_issue / (1 - r_cashout)
P_floor* = (1 - phi_tot) * P_issue
```

The steady floor is **independent** of the cash-out tax!

### Floor Dynamics

The floor growth rate is:

```
dP_floor/dt = [(1-r_cashout)/S] * [r_in * (S - B/P_issue) + r_cashout * B * r_out]
```

Two channels:
1. **Issuance channel**: Positive when `P_issue > B/S` (price above backing)
2. **Redemption channel**: Always positive when `r_cashout > 0` and `r_out > 0`

### Neutral Line

The floor is stationary when:

```
r_in * (S - B/P_issue) + r_cashout * B * r_out = 0
```

## Examples

Run the comprehensive example suite:

```bash
python examples.py
```

This generates 7 examples demonstrating all features:

1. **Basic Simulation** - ODE integration with constant rates
2. **Analytical Comparison** - Validate numerics against closed-form solutions
3. **Counterfactual Analysis** - Compare multiple rate scenarios
4. **Parameter Sweep** - Explore effect of cash-out tax
5. **Uncertainty Quantification** - Monte Carlo with uncertain rates
6. **Sensitivity Analysis** - Latin Hypercube Sampling and Sobol indices
7. **Agent-Based Model** - Heterogeneous agents with emergent dynamics

All plots are saved to the outputs directory.

## Advanced Usage

### Custom Agent Strategies

```python
from agent_based_model import Agent, AgentType

class MyCustomAgent(Agent):
    def compute_action(self, t, dt, P_issue, P_floor):
        # Custom logic here
        cash_in = ...
        redeem = ...
        return cash_in, redeem
```

### Custom Rate Functions

```python
def time_varying_rate(t):
    """Custom rate function"""
    if t < 50:
        return 50.0 + 0.5 * t
    else:
        return 75.0 - 0.3 * (t - 50)

sim.simulate(t_end=100, r_in_func=time_varying_rate, r_out_func=constant_rate(40))
```

### Multi-Stage Simulations

```python
stages = [
    Stage(t_start=0.0, P_issue_0=1.0, gamma_cut=0.05, Delta_t=10, r_cashout=0.1),
    Stage(t_start=100.0, P_issue_0=1.5, gamma_cut=0.03, Delta_t=15, r_cashout=0.15),
    Stage(t_start=200.0, P_issue_0=2.0, gamma_cut=0.02, Delta_t=20, r_cashout=0.2),
]

sim = RevnetSimulator(stages, initial_state, phi_tot)
sim.simulate(t_end=300, r_in_func, r_out_func)
```

## Performance Tips

- Use larger `dt` for faster simulations (but check accuracy)
- For long simulations, consider breaking into intervals
- ABM simulations are slower; use smaller agent populations for testing
- Use analytical solutions when possible for validation

## Limitations

- No support for external market prices (assumes mechanism is primary market)
- Agent strategies are stylized (not calibrated to real behavior)
- Monte Carlo UQ assumes independent samples (no correlation structure)
- Sobol indices are first-order approximations

## Citation

If you use this simulator in research, please cite:

```
The CEL Team. "Revnet Value Flows as a Continuous-Time Dynamical System:
A Rate-Based Formalization." 2025.
```

## License

MIT License

## Contact

For questions or contributions, please open an issue or submit a pull request.
