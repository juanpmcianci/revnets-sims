"""
Comprehensive Example: Revnet Simulation
=========================================

This script demonstrates all features of the simulation environment:
1. Basic ODE simulation with constant rates
2. Agent-based modeling
3. Counterfactual analysis
4. Uncertainty quantification
5. Sensitivity analysis
6. Parameter sweeps
"""

import numpy as np
import matplotlib.pyplot as plt

from revnet_simulator import (
    Stage, RevnetState, RevnetSimulator,
    constant_rate, piecewise_rate, sinusoidal_rate
)
from agent_based_model import (
    AgentBasedRevnetSimulator, Agent, AgentType, create_agent_population
)
from analysis import (
    AnalyticalSolver, CounterfactualAnalyzer, UncertaintyQuantifier,
    SensitivityAnalyzer, SteadyState
)
from visualization import (
    plot_simulation_results, plot_counterfactual_comparison,
    plot_uncertainty_quantification, plot_parameter_sweep,
    plot_sensitivity_analysis, plot_lhs_results, plot_agent_behavior
)


def example_1_basic_simulation():
    """
    Example 1: Basic ODE simulation with constant rates.
    """
    print("=" * 60)
    print("Example 1: Basic ODE Simulation")
    print("=" * 60)
    
    # Define a single stage
    stage = Stage(
        t_start=0.0,
        P_issue_0=1.0,
        gamma_cut=0.05,  # 5% price cut -> gamma = 1/0.95 ≈ 1.053
        Delta_t=10.0,    # Price steps every 10 time units
        sigma=0.1,
        r_cashout=0.1,   # 10% cash-out tax
        auto_issuances=[(50.0, 100.0)]  # Auto-issue 100 tokens at t=50
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
    
    # Define constant rate functions
    r_in = constant_rate(50.0)   # 50 base units per time
    r_out = constant_rate(20.0)  # 20 tokens per time
    
    # Run simulation
    print("Running simulation to t=100...")
    sim.simulate(t_end=100.0, r_in_func=r_in, r_out_func=r_out, dt=0.1)
    
    # Get results
    history = sim.get_history()
    
    print(f"Initial floor: {history['P_floor'][0]:.4f}")
    print(f"Final floor: {history['P_floor'][-1]:.4f}")
    print(f"Floor growth: {(history['P_floor'][-1] / history['P_floor'][0] - 1) * 100:.2f}%")
    print(f"Initial backing ratio: {history['B'][0] / history['S'][0]:.4f}")
    print(f"Final backing ratio: {history['B'][-1] / history['S'][-1]:.4f}")
    
    # Plot results
    fig = plot_simulation_results(history, title="Example 1: Basic Simulation")
    plt.savefig('./results/example1_basic_simulation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to example1_basic_simulation.png")
    print()


def example_2_analytical_solutions():
    """
    Example 2: Compare numerical and analytical solutions.
    """
    print("=" * 60)
    print("Example 2: Analytical vs Numerical Solutions")
    print("=" * 60)
    
    # Parameters
    B_0, S_0 = 1000.0, 1000.0
    r_in, r_out = 50.0, 40.0
    P_issue = 1.0
    r_cashout = 0.1
    t_end = 50.0
    
    # Analytical solution
    print("Computing analytical solution...")
    t_points = np.linspace(0, t_end, 500)
    S_analytical = []
    B_analytical = []
    
    for t in t_points:
        S_t, B_t = AnalyticalSolver.solve_constant_rates(
            B_0, S_0, r_in, r_out, P_issue, r_cashout, t
        )
        S_analytical.append(S_t)
        B_analytical.append(B_t)
    
    S_analytical = np.array(S_analytical)
    B_analytical = np.array(B_analytical)
    
    # Numerical solution
    print("Computing numerical solution...")
    stage = Stage(t_start=0.0, P_issue_0=P_issue, gamma_cut=0.0, Delta_t=1000.0, r_cashout=r_cashout)
    sim = RevnetSimulator(
        stages=[stage],
        initial_state=RevnetState(B_0, S_0, 0.0),
        phi_tot=0.0,
        seed=42
    )
    sim.simulate(t_end, constant_rate(r_in), constant_rate(r_out), dt=0.1)
    history = sim.get_history()
    
    # Compare
    S_numerical = history['S']
    B_numerical = history['B']
    t_numerical = history['t']
    
    # Interpolate for comparison
    S_num_interp = np.interp(t_points, t_numerical, S_numerical)
    B_num_interp = np.interp(t_points, t_numerical, B_numerical)
    
    S_error = np.abs(S_analytical - S_num_interp) / S_analytical
    B_error = np.abs(B_analytical - B_num_interp) / B_analytical
    
    print(f"Max relative error in S: {np.max(S_error) * 100:.4f}%")
    print(f"Max relative error in B: {np.max(B_error) * 100:.4f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(t_points, S_analytical, 'b-', linewidth=2, label='Analytical')
    axes[0, 0].plot(t_numerical, S_numerical, 'r--', linewidth=1.5, label='Numerical')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Token Supply (S)')
    axes[0, 0].set_title('Token Supply Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_points, B_analytical, 'b-', linewidth=2, label='Analytical')
    axes[0, 1].plot(t_numerical, B_numerical, 'r--', linewidth=1.5, label='Numerical')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Treasury (B)')
    axes[0, 1].set_title('Treasury Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].semilogy(t_points, S_error, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Relative Error')
    axes[1, 0].set_title('Relative Error in S')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(t_points, B_error, 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Relative Error')
    axes[1, 1].set_title('Relative Error in B')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Analytical vs Numerical Solutions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/example2_analytical_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to example2_analytical_comparison.png")
    print()


def example_3_counterfactual_analysis():
    """
    Example 3: Counterfactual analysis with different rate scenarios.
    """
    print("=" * 60)
    print("Example 3: Counterfactual Analysis")
    print("=" * 60)
    
    stage = Stage(
        t_start=0.0,
        P_issue_0=1.0,
        gamma_cut=0.05,
        Delta_t=20.0,
        r_cashout=0.15
    )
    
    initial_state = RevnetState(B=1000.0, S=1000.0, t=0.0)
    
    # Define scenarios
    scenarios = {
        "High Issuance": (constant_rate(100.0), constant_rate(30.0)),
        "Balanced": (constant_rate(50.0), constant_rate(50.0)),
        "High Redemption": (constant_rate(30.0), constant_rate(60.0)),
        "Cyclical": (
            sinusoidal_rate(50.0, 30.0, 40.0),
            sinusoidal_rate(40.0, 20.0, 40.0, phase=np.pi)
        ),
    }
    
    analyzer = CounterfactualAnalyzer(
        stages=[stage],
        initial_state=initial_state,
        phi_tot=0.02,
        seed=42
    )
    
    print("Running counterfactual scenarios...")
    results = analyzer.compare_rate_scenarios(
        t_end=100.0,
        rate_scenarios=scenarios,
        dt=0.1
    )
    
    # Print summary
    print("\nScenario Comparison:")
    print("-" * 60)
    for scenario_name, result in results.items():
        print(f"{scenario_name:20s} | Floor: {result['final_floor']:.4f} | "
              f"Growth: {result['floor_growth']*100:+.2f}%")
    
    # Plot
    fig = plot_counterfactual_comparison(
        results,
        title="Counterfactual Analysis: Rate Scenarios"
    )
    plt.savefig('results/example3_counterfactual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to example3_counterfactual.png")
    print()


def example_4_parameter_sweep():
    """
    Example 4: Parameter sweep over cash-out tax.
    """
    print("=" * 60)
    print("Example 4: Parameter Sweep (Cash-Out Tax)")
    print("=" * 60)
    
    stage = Stage(
        t_start=0.0,
        P_issue_0=1.0,
        gamma_cut=0.05,
        Delta_t=20.0,
        r_cashout=0.1  # Will be varied
    )
    
    initial_state = RevnetState(B=1000.0, S=1000.0, t=0.0)
    
    analyzer = CounterfactualAnalyzer(
        stages=[stage],
        initial_state=initial_state,
        phi_tot=0.02,
        seed=42
    )
    
    # Sweep cash-out tax from 0 to 0.5
    r_cashout_values = np.linspace(0.0, 0.5, 11)
    
    print(f"Sweeping r_cashout from {r_cashout_values[0]} to {r_cashout_values[-1]}...")
    
    results = analyzer.parameter_sweep(
        t_end=100.0,
        param_name='r_cashout',
        param_values=r_cashout_values,
        r_in_func=constant_rate(50.0),
        r_out_func=constant_rate(40.0),
        dt=0.1
    )
    
    # Print results
    print("\nParameter Sweep Results:")
    print("-" * 60)
    for r_cashout, result in results.items():
        print(f"r_cashout={r_cashout:.2f} | Final Floor: {result['final_floor']:.4f} | "
              f"Avg Floor: {result['avg_floor']:.4f}")
    
    # Plot
    fig = plot_parameter_sweep(
        results,
        param_name='r_cashout',
        title="Parameter Sweep: Cash-Out Tax Effect"
    )
    plt.savefig('./results/example4_parameter_sweep.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to example4_parameter_sweep.png")
    print()


def example_5_uncertainty_quantification():
    """
    Example 5: Monte Carlo uncertainty quantification.
    """
    print("=" * 60)
    print("Example 5: Uncertainty Quantification")
    print("=" * 60)
    
    stage = Stage(
        t_start=0.0,
        P_issue_0=1.0,
        gamma_cut=0.05,
        Delta_t=20.0,
        r_cashout=0.1
    )
    
    initial_state = RevnetState(B=1000.0, S=1000.0, t=0.0)
    
    # Define uncertain rate samplers
    def r_in_sampler(rng):
        """Sample r_in from normal distribution"""
        mean_rate = 50.0
        std_rate = 10.0
        rate = max(0, rng.normal(mean_rate, std_rate))
        return constant_rate(rate)
    
    def r_out_sampler(rng):
        """Sample r_out from normal distribution"""
        mean_rate = 40.0
        std_rate = 8.0
        rate = max(0, rng.normal(mean_rate, std_rate))
        return constant_rate(rate)
    
    quantifier = UncertaintyQuantifier(
        stages=[stage],
        initial_state=initial_state,
        phi_tot=0.02,
        seed=42
    )
    
    print("Running Monte Carlo simulations (n=100)...")
    uq_results = quantifier.monte_carlo_rates(
        t_end=100.0,
        r_in_sampler=r_in_sampler,
        r_out_sampler=r_out_sampler,
        n_samples=100,
        dt=0.1
    )
    
    # Print statistics
    print("\nUncertainty Statistics (Final Time):")
    print("-" * 60)
    print(f"Floor Price:")
    print(f"  Mean: {uq_results['P_floor_mean'][-1]:.4f}")
    print(f"  Std:  {uq_results['P_floor_std'][-1]:.4f}")
    print(f"  5%:   {uq_results['P_floor_q05'][-1]:.4f}")
    print(f"  95%:  {uq_results['P_floor_q95'][-1]:.4f}")
    print(f"Treasury:")
    print(f"  Mean: {uq_results['B_mean'][-1]:.2f}")
    print(f"  Std:  {uq_results['B_std'][-1]:.2f}")
    
    # Plot
    fig = plot_uncertainty_quantification(
        uq_results,
        title="Uncertainty Quantification: Monte Carlo (n=100)"
    )
    plt.savefig('example5_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to example5_uncertainty.png")
    print()


def example_6_sensitivity_analysis():
    """
    Example 6: Sensitivity analysis using Latin Hypercube Sampling.
    """
    print("=" * 60)
    print("Example 6: Sensitivity Analysis")
    print("=" * 60)
    
    stage = Stage(
        t_start=0.0,
        P_issue_0=1.0,
        gamma_cut=0.05,
        Delta_t=20.0,
        r_cashout=0.1
    )
    
    initial_state = RevnetState(B=1000.0, S=1000.0, t=0.0)
    
    quantifier = UncertaintyQuantifier(
        stages=[stage],
        initial_state=initial_state,
        phi_tot=0.02,
        seed=42
    )
    
    # Define parameter distributions
    param_distributions = {
        'r_cashout': (0.0, 0.3),
        'gamma_cut': (0.01, 0.10),
        'r_in': (30.0, 70.0),
        'r_out': (20.0, 60.0),
    }
    
    print("Running Latin Hypercube Sampling (n=200)...")
    lhs_results = quantifier.latin_hypercube_sampling(
        t_end=100.0,
        param_distributions=param_distributions,
        base_r_in=50.0,
        base_r_out=40.0,
        n_samples=200,
        dt=0.1
    )
    
    # Compute Sobol indices
    print("\nComputing sensitivity indices...")
    sobol_indices = SensitivityAnalyzer.compute_sobol_indices(
        lhs_results,
        outcome_key='final_floor'
    )
    
    print("\nSobol Indices (Approximation):")
    print("-" * 60)
    for param, index in sorted(sobol_indices.items(), key=lambda x: -x[1]):
        print(f"{param:15s}: {index:.4f}")
    
    # Plot sensitivity
    fig1 = plot_sensitivity_analysis(
        sobol_indices,
        title="Sensitivity Analysis: Effect on Final Floor"
    )
    plt.savefig('./results/example6_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot LHS scatter
    fig2 = plot_lhs_results(
        lhs_results,
        x_param='r_cashout',
        y_param='gamma_cut',
        outcome_key='final_floor',
        title="LHS: Parameter Space Exploration"
    )
    plt.savefig('./results/example6_lhs_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots saved to example6_sensitivity.png and example6_lhs_scatter.png")
    print()


def example_7_agent_based_model():
    """
    Example 7: Agent-based simulation with heterogeneous agents.
    """
    print("=" * 60)
    print("Example 7: Agent-Based Model")
    print("=" * 60)
    
    stage = Stage(
        t_start=0.0,
        P_issue_0=1.0,
        gamma_cut=0.05,
        Delta_t=20.0,
        r_cashout=0.1
    )
    
    initial_state = RevnetState(B=1000.0, S=1000.0, t=0.0)
    
    # Create heterogeneous agent population
    agent_distribution = {
        AgentType.RANDOM: 0.20,
        AgentType.PRICE_SENSITIVE: 0.25,
        AgentType.FLOOR_TRADER: 0.20,
        AgentType.MOMENTUM: 0.15,
        AgentType.HODLER: 0.15,
        AgentType.ARBITRAGEUR: 0.05,
    }
    
    print("Creating agent population (n=100)...")
    agents = create_agent_population(
        n_agents=100,
        agent_type_distribution=agent_distribution,
        seed=42
    )
    
    # Create ABM simulator
    abm_sim = AgentBasedRevnetSimulator(
        stages=[stage],
        initial_state=initial_state,
        agents=agents,
        phi_tot=0.02,
        seed=42
    )
    
    print("Running agent-based simulation to t=100...")
    abm_sim.simulate(t_end=100.0, dt=0.5)
    
    # Print results
    print(f"\nFinal state:")
    print(f"  Treasury: {abm_sim.state.B:.2f}")
    print(f"  Supply: {abm_sim.state.S:.2f}")
    print(f"  Floor: {abm_sim.state.marginal_floor(stage.r_cashout, abm_sim.phi_tot):.4f}")
    print(f"  Total transactions: {len(abm_sim.transactions)}")
    
    # Plot agent behavior
    fig = plot_agent_behavior(
        abm_sim,
        title="Agent-Based Model: Aggregate Flows"
    )
    plt.savefig('./results/example7_agent_based.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to example7_agent_based.png")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("REVNET SIMULATION EXAMPLES")
    print("=" * 60 + "\n")
    
    # Run examples
    example_1_basic_simulation()
    example_2_analytical_solutions()
    example_3_counterfactual_analysis()
    example_4_parameter_sweep()
    example_5_uncertainty_quantification()
    example_6_sensitivity_analysis()
    example_7_agent_based_model()
    
    print("=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print("\nAll plots saved to ./results/")


if __name__ == "__main__":
    main()
