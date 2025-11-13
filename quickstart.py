"""
Revnet Simulator - Quick Start Guide
=====================================

This tutorial walks through the basic usage of the Revnet simulator.
"""

import numpy as np
import matplotlib.pyplot as plt

from revnet_simulator import Stage, RevnetState, RevnetSimulator, constant_rate
from visualization import plot_simulation_results

print("=" * 70)
print("REVNET SIMULATOR - QUICK START")
print("=" * 70)
print()

# =============================================================================
# PART 1: Your First Simulation
# =============================================================================

print("PART 1: Your First Simulation")
print("-" * 70)

# Step 1: Define a stage
# A stage defines the economic parameters for a period of time
stage = Stage(
    t_start=0.0,         # When this stage begins
    P_issue_0=1.0,       # Initial issuance price (1.0 base units per token)
    gamma_cut=0.05,      # Price cut factor: 5% per step
    Delta_t=10.0,        # Time between price steps (10 time units)
    sigma=0.1,           # Per-mint split parameter
    r_cashout=0.1,       # Cash-out tax: 10%
)

print(f"✓ Created stage with:")
print(f"  - Initial price: {stage.P_issue_0}")
print(f"  - Cash-out tax: {stage.r_cashout * 100}%")
print(f"  - Price increases by factor {stage.gamma:.3f} every {stage.Delta_t} time units")

# Step 2: Set initial conditions
# B = Treasury balance (base asset)
# S = Token supply
initial_state = RevnetState(
    B=1000.0,  # Start with 1000 base units in treasury
    S=1000.0,  # Start with 1000 tokens in circulation
    t=0.0      # Start at time 0
)

print(f"\n✓ Initial state:")
print(f"  - Treasury (B): {initial_state.B}")
print(f"  - Supply (S): {initial_state.S}")
print(f"  - Initial backing ratio: {initial_state.backing_ratio():.4f}")

# Step 3: Create the simulator
sim = RevnetSimulator(
    stages=[stage],
    initial_state=initial_state,
    phi_tot=0.02,  # 2% protocol fee on redemptions
    seed=42        # For reproducibility
)

print(f"\n✓ Created simulator with {len(sim.stages)} stage(s)")

# Step 4: Define activity rates
# These represent expected user behavior
r_in = constant_rate(50.0)   # 50 base units per time flowing in (issuance)
r_out = constant_rate(40.0)  # 40 tokens per time flowing out (redemption)

print(f"\n✓ Defined activity rates:")
print(f"  - Cash-in rate: 50.0 base units/time")
print(f"  - Cash-out rate: 40.0 tokens/time")

# Step 5: Run the simulation
print(f"\nRunning simulation to t=100...")
sim.simulate(
    t_end=100.0,
    r_in_func=r_in,
    r_out_func=r_out,
    dt=0.1  # Time step for numerical integration
)

print("✓ Simulation complete!")

# Step 6: Get and analyze results
history = sim.get_history()

print(f"\nResults:")
print(f"  Initial floor price: {history['P_floor'][0]:.4f}")
print(f"  Final floor price:   {history['P_floor'][-1]:.4f}")
print(f"  Floor growth:        {(history['P_floor'][-1]/history['P_floor'][0] - 1)*100:.2f}%")
print(f"  Final treasury:      {history['B'][-1]:.2f}")
print(f"  Final supply:        {history['S'][-1]:.2f}")
print(f"  Final backing ratio: {history['B'][-1]/history['S'][-1]:.4f}")

print()

# =============================================================================
# PART 2: Understanding the Dynamics
# =============================================================================

print("\nPART 2: Understanding the Dynamics")
print("-" * 70)

# The floor price is determined by the backing ratio and parameters:
# P_floor = (1 - phi_tot) * (1 - r_cashout) * (B / S)

current_backing = history['B'][-1] / history['S'][-1]
current_floor = history['P_floor'][-1]
expected_floor = (1 - 0.02) * (1 - 0.1) * current_backing

print(f"Floor price calculation:")
print(f"  Backing ratio (B/S):     {current_backing:.4f}")
print(f"  (1 - phi_tot):           {1 - 0.02:.4f}")
print(f"  (1 - r_cashout):         {1 - 0.1:.4f}")
print(f"  Expected floor:          {expected_floor:.4f}")
print(f"  Actual floor:            {current_floor:.4f}")
print(f"  Match: {'✓' if abs(expected_floor - current_floor) < 0.001 else '✗'}")

# The floor grows when:
# 1. Issuance at price above current backing (P_issue > B/S)
# 2. Redemptions occur with positive cash-out tax (r_out > 0 and r_cashout > 0)

print(f"\nFloor growth mechanisms:")
print(f"  1. Issuance channel: Active when P_issue > B/S")
print(f"  2. Redemption channel: Active when r_out > 0 and r_cashout > 0")
print(f"     (Burns tokens while retaining value in treasury)")

print()

# =============================================================================
# PART 3: Comparing Scenarios
# =============================================================================

print("\nPART 3: Comparing Scenarios")
print("-" * 70)

scenarios = {
    "Baseline": (constant_rate(50.0), constant_rate(40.0)),
    "High Growth": (constant_rate(80.0), constant_rate(30.0)),
    "Steady State": (constant_rate(50.0), constant_rate(50.0)),
}

results = {}
for name, (r_in_func, r_out_func) in scenarios.items():
    sim_scenario = RevnetSimulator(
        stages=[stage],
        initial_state=RevnetState(1000.0, 1000.0, 0.0),
        phi_tot=0.02,
        seed=42
    )
    sim_scenario.simulate(100.0, r_in_func, r_out_func, 0.1)
    hist = sim_scenario.get_history()
    results[name] = {
        'final_floor': hist['P_floor'][-1],
        'floor_growth': (hist['P_floor'][-1] / hist['P_floor'][0] - 1) * 100
    }

print("Scenario comparison:")
for name, result in results.items():
    print(f"  {name:15s} | Floor: {result['final_floor']:.4f} | Growth: {result['floor_growth']:+.2f}%")

print()

# =============================================================================
# PART 4: Key Insights
# =============================================================================

print("\nPART 4: Key Insights from Your First Simulation")
print("-" * 70)

print("""
1. FLOOR DYNAMICS
   - The floor price tracks the backing ratio (B/S)
   - Floor grows through two channels: issuance and redemption
   - At steady state: P_floor ≈ (1 - phi_tot) * P_issue

2. PARAMETER EFFECTS
   - Higher r_cashout → More value retained during redemptions
   - Higher r_in/r_out → Faster dynamics
   - Price steps (gamma) affect issuance channel strength

3. STEADY STATE
   - Balanced flows (r_in/P_issue = r_out) reach equilibrium
   - Steady floor = (1 - phi_tot) * P_issue (independent of r_cashout!)
   - Backing ratio = P_issue / (1 - r_cashout)

4. NEXT STEPS
   - Try the full examples.py for advanced features
   - Explore agent-based modeling for emergent behavior
   - Use counterfactual analysis for design decisions
   - Run uncertainty quantification for robust design
""")

print("=" * 70)
print("Tutorial complete! See README.md and examples.py for more.")
print("=" * 70)

# Save a simple plot
fig = plot_simulation_results(history, title="Your First Revnet Simulation")
plt.savefig('./results/quickstart_simulation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Plot saved to quickstart_simulation.png")
