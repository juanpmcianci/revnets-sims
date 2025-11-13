"""
Test Script - Verify Revnet Simulator Installation
===================================================

Run this script to verify that the simulator is working correctly.
"""

import sys

print("=" * 70)
print("REVNET SIMULATOR - INSTALLATION TEST")
print("=" * 70)
print()

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    import numpy as np
    import scipy
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✓ All dependencies imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("  Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Import simulator modules
print("\nTest 2: Importing simulator modules...")
try:
    from revnet_simulator import Stage, RevnetState, RevnetSimulator, constant_rate
    from agent_based_model import AgentBasedRevnetSimulator, AgentType, create_agent_population
    from analysis import AnalyticalSolver, CounterfactualAnalyzer, UncertaintyQuantifier
    from visualization import plot_simulation_results
    print("✓ All simulator modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("  Make sure all .py files are in the same directory")
    sys.exit(1)

# Test 3: Create a stage
print("\nTest 3: Creating a stage...")
try:
    stage = Stage(
        t_start=0.0,
        P_issue_0=1.0,
        gamma_cut=0.05,
        Delta_t=10.0,
        r_cashout=0.1
    )
    assert stage.gamma > 1.0, "Gamma should be greater than 1"
    print(f"✓ Stage created with gamma = {stage.gamma:.4f}")
except Exception as e:
    print(f"✗ Stage creation failed: {e}")
    sys.exit(1)

# Test 4: Run a basic simulation
print("\nTest 4: Running basic simulation...")
try:
    initial_state = RevnetState(B=1000.0, S=1000.0, t=0.0)
    sim = RevnetSimulator([stage], initial_state, phi_tot=0.02, seed=42)
    sim.simulate(t_end=10.0, r_in_func=constant_rate(50.0), r_out_func=constant_rate(40.0), dt=0.1)
    history = sim.get_history()
    
    assert len(history['t']) > 1, "Should have multiple time points"
    assert history['B'][-1] > 0, "Treasury should be positive"
    assert history['S'][-1] > 0, "Supply should be positive"
    
    print(f"✓ Simulation ran successfully")
    print(f"  Simulated {len(history['t'])} time points")
    print(f"  Final treasury: {history['B'][-1]:.2f}")
    print(f"  Final supply: {history['S'][-1]:.2f}")
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test analytical solver
print("\nTest 5: Testing analytical solver...")
try:
    S_t, B_t = AnalyticalSolver.solve_constant_rates(
        B_0=1000.0, S_0=1000.0, r_in=50.0, r_out=40.0,
        P_issue=1.0, r_cashout=0.1, t=10.0
    )
    assert S_t > 0, "Supply should be positive"
    assert B_t > 0, "Treasury should be positive"
    print(f"✓ Analytical solution computed")
    print(f"  S(10) = {S_t:.2f}")
    print(f"  B(10) = {B_t:.2f}")
except Exception as e:
    print(f"✗ Analytical solver failed: {e}")
    sys.exit(1)

# Test 6: Test agent population creation
print("\nTest 6: Creating agent population...")
try:
    agents = create_agent_population(
        n_agents=10,
        agent_type_distribution={
            AgentType.RANDOM: 0.5,
            AgentType.HODLER: 0.5,
        },
        seed=42
    )
    assert len(agents) == 10, "Should have 10 agents"
    print(f"✓ Agent population created with {len(agents)} agents")
except Exception as e:
    print(f"✗ Agent creation failed: {e}")
    sys.exit(1)

# Test 7: Test visualization (no display)
print("\nTest 7: Testing visualization...")
try:
    plt.ioff()  # Turn off interactive mode
    fig = plot_simulation_results(history, title="Test Plot")
    plt.close(fig)
    print("✓ Visualization functions work")
except Exception as e:
    print(f"✗ Visualization failed: {e}")
    sys.exit(1)

# All tests passed
print()
print("=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print()
print("The Revnet simulator is installed correctly and working!")
print("You can now:")
print("  - Run quickstart.py for a tutorial")
print("  - Run examples.py for comprehensive examples")
print("  - Import modules in your own scripts")
print()
