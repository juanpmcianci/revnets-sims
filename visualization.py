"""
Visualization Tools for Revnet Simulations
===========================================

Plotting functions for:
- Time series of state variables
- Phase plots
- Counterfactual comparisons
- Uncertainty bands
- Sensitivity analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_simulation_results(
    history: Dict[str, np.ndarray],
    title: str = "Revnet Simulation",
    save_path: Optional[str] = None
):
    """
    Plot main simulation results: B, S, P_floor, P_issue over time.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    t = history['t']
    
    # Treasury balance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, history['B'], 'b-', linewidth=2, label='B(t)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Treasury Balance (B)')
    ax1.set_title('Treasury Balance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Token supply
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, history['S'], 'g-', linewidth=2, label='S(t)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Token Supply (S)')
    ax2.set_title('Token Supply')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Backing ratio
    ax3 = fig.add_subplot(gs[1, 0])
    backing_ratio = history['B'] / history['S']
    ax3.plot(t, backing_ratio, 'purple', linewidth=2, label='B/S')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Backing Ratio (B/S)')
    ax3.set_title('Backing Ratio')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Floor and issuance prices
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, history['P_floor'], 'r-', linewidth=2, label='Floor Price')
    ax4.plot(t, history['P_issue'], 'k--', linewidth=2, alpha=0.7, label='Issuance Price')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.set_title('Floor vs Issuance Price')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Spread (P_issue - P_floor)
    ax5 = fig.add_subplot(gs[2, 0])
    spread = history['P_issue'] - history['P_floor']
    ax5.plot(t, spread, 'orange', linewidth=2)
    ax5.fill_between(t, 0, spread, alpha=0.3, color='orange')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Price Spread')
    ax5.set_title('Issuance-Floor Spread')
    ax5.grid(True, alpha=0.3)
    
    # Phase plot: B vs S
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(history['S'], history['B'], 'b-', linewidth=1.5, alpha=0.7)
    ax6.scatter(history['S'][0], history['B'][0], c='green', s=100, marker='o', 
                label='Start', zorder=5)
    ax6.scatter(history['S'][-1], history['B'][-1], c='red', s=100, marker='s', 
                label='End', zorder=5)
    ax6.set_xlabel('Token Supply (S)')
    ax6.set_ylabel('Treasury Balance (B)')
    ax6.set_title('Phase Plot: Treasury vs Supply')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_counterfactual_comparison(
    results: Dict[str, Dict],
    metric: str = 'P_floor',
    title: str = "Counterfactual Analysis",
    save_path: Optional[str] = None
):
    """
    Compare multiple scenarios on a single plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Define metrics to plot
    metrics = [
        ('P_floor', 'Floor Price', axes[0, 0]),
        ('B', 'Treasury Balance', axes[0, 1]),
        ('S', 'Token Supply', axes[1, 0]),
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (metric_key, ylabel, ax) in metrics:
        for i, (scenario_name, result) in enumerate(results.items()):
            history = result['history']
            t = history['t']
            y = history[metric_key]
            ax.plot(t, y, linewidth=2, label=scenario_name, color=colors[i])
        
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Summary table in fourth subplot
    ax_table = axes[1, 1]
    ax_table.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Scenario', 'Final Floor', 'Floor Growth', 'Treasury Growth']
    
    for scenario_name, result in results.items():
        row = [
            scenario_name,
            f"{result['final_floor']:.4f}",
            f"{result['floor_growth']*100:.2f}%",
            f"{result['treasury_growth']*100:.2f}%"
        ]
        table_data.append(row)
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_uncertainty_quantification(
    uq_results: Dict[str, np.ndarray],
    title: str = "Uncertainty Quantification",
    save_path: Optional[str] = None
):
    """
    Plot uncertainty bands from Monte Carlo simulations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t = uq_results['t']
    
    # Treasury balance with uncertainty
    ax1 = axes[0, 0]
    ax1.plot(t, uq_results['B_mean'], 'b-', linewidth=2, label='Mean')
    ax1.fill_between(
        t, 
        uq_results['B_q05'], 
        uq_results['B_q95'],
        alpha=0.3,
        color='blue',
        label='90% CI'
    )
    ax1.fill_between(
        t,
        uq_results['B_mean'] - uq_results['B_std'],
        uq_results['B_mean'] + uq_results['B_std'],
        alpha=0.2,
        color='blue',
        label='±1 std'
    )
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Treasury Balance (B)')
    ax1.set_title('Treasury Balance - Uncertainty')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Floor price with uncertainty
    ax2 = axes[0, 1]
    ax2.plot(t, uq_results['P_floor_mean'], 'r-', linewidth=2, label='Mean')
    ax2.fill_between(
        t,
        uq_results['P_floor_q05'],
        uq_results['P_floor_q95'],
        alpha=0.3,
        color='red',
        label='90% CI'
    )
    ax2.fill_between(
        t,
        uq_results['P_floor_mean'] - uq_results['P_floor_std'],
        uq_results['P_floor_mean'] + uq_results['P_floor_std'],
        alpha=0.2,
        color='red',
        label='±1 std'
    )
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Floor Price')
    ax2.set_title('Floor Price - Uncertainty')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # All trajectories for B (sample)
    ax3 = axes[1, 0]
    n_samples = min(50, uq_results['B'].shape[0])
    for i in range(n_samples):
        ax3.plot(t, uq_results['B'][i, :], alpha=0.1, color='blue')
    ax3.plot(t, uq_results['B_mean'], 'b-', linewidth=2, label='Mean')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Treasury Balance (B)')
    ax3.set_title(f'Sample Trajectories (n={n_samples})')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # All trajectories for P_floor (sample)
    ax4 = axes[1, 1]
    for i in range(n_samples):
        ax4.plot(t, uq_results['P_floor'][i, :], alpha=0.1, color='red')
    ax4.plot(t, uq_results['P_floor_mean'], 'r-', linewidth=2, label='Mean')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Floor Price')
    ax4.set_title(f'Sample Trajectories (n={n_samples})')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_parameter_sweep(
    sweep_results: Dict[float, Dict],
    param_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize results of parameter sweep.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    param_values = sorted(sweep_results.keys())
    
    # Extract metrics
    final_floors = [sweep_results[p]['final_floor'] for p in param_values]
    final_ratios = [sweep_results[p]['final_backing_ratio'] for p in param_values]
    avg_floors = [sweep_results[p]['avg_floor'] for p in param_values]
    
    # Final floor vs parameter
    ax1 = axes[0, 0]
    ax1.plot(param_values, final_floors, 'o-', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Final Floor Price')
    ax1.set_title(f'Final Floor vs {param_name}')
    ax1.grid(True, alpha=0.3)
    
    # Final backing ratio vs parameter
    ax2 = axes[0, 1]
    ax2.plot(param_values, final_ratios, 's-', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Final Backing Ratio (B/S)')
    ax2.set_title(f'Final Backing Ratio vs {param_name}')
    ax2.grid(True, alpha=0.3)
    
    # Average floor vs parameter
    ax3 = axes[1, 0]
    ax3.plot(param_values, avg_floors, '^-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Average Floor Price')
    ax3.set_title(f'Average Floor vs {param_name}')
    ax3.grid(True, alpha=0.3)
    
    # Floor trajectories for selected parameters
    ax4 = axes[1, 1]
    n_show = min(5, len(param_values))
    indices = np.linspace(0, len(param_values)-1, n_show, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_show))
    
    for i, idx in enumerate(indices):
        p = param_values[idx]
        history = sweep_results[p]['history']
        ax4.plot(
            history['t'], 
            history['P_floor'], 
            linewidth=2, 
            color=colors[i],
            label=f'{param_name}={p:.3f}'
        )
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Floor Price')
    ax4.set_title('Floor Trajectories (Selected Parameters)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    if title is None:
        title = f"Parameter Sweep: {param_name}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_sensitivity_analysis(
    sensitivity_results: Dict[str, float],
    title: str = "Sensitivity Analysis",
    save_path: Optional[str] = None
):
    """
    Bar plot of sensitivity indices.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    params = list(sensitivity_results.keys())
    indices = list(sensitivity_results.values())
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(params)))
    bars = ax.barh(params, indices, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Sensitivity Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, indices)):
        ax.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_lhs_results(
    lhs_results: Dict,
    x_param: str,
    y_param: str,
    outcome_key: str = 'final_floor',
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Scatter plot of LHS results colored by outcome.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    param_names = lhs_results['param_names']
    param_samples = lhs_results['param_samples']
    outcomes = np.array([o[outcome_key] for o in lhs_results['outcomes']])
    
    x_idx = param_names.index(x_param)
    y_idx = param_names.index(y_param)
    
    x_vals = param_samples[:, x_idx]
    y_vals = param_samples[:, y_idx]
    
    # Scatter plot with color
    ax1 = axes[0]
    scatter = ax1.scatter(
        x_vals, 
        y_vals, 
        c=outcomes, 
        cmap='viridis', 
        s=50, 
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax1.set_xlabel(x_param, fontsize=12)
    ax1.set_ylabel(y_param, fontsize=12)
    ax1.set_title(f'LHS: {x_param} vs {y_param}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label=outcome_key)
    
    # Distribution of outcomes
    ax2 = axes[1]
    # Filter out inf and nan values
    outcomes_finite = outcomes[np.isfinite(outcomes)]
    if len(outcomes_finite) > 0:
        ax2.hist(outcomes_finite, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(outcomes_finite), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(np.median(outcomes_finite), color='green', linestyle='--', linewidth=2, label='Median')
        ax2.set_xlabel(outcome_key, fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Distribution of {outcome_key}', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # Add text for inf/nan count
        n_inf = np.sum(np.isinf(outcomes))
        n_nan = np.sum(np.isnan(outcomes))
        if n_inf > 0 or n_nan > 0:
            ax2.text(0.98, 0.98, f'Inf: {n_inf}, NaN: {n_nan}', 
                    transform=ax2.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No finite outcomes', 
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=14, color='red')
    
    if title is None:
        title = "Latin Hypercube Sampling Results"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_agent_behavior(
    abm_simulator,
    title: str = "Agent-Based Model Results",
    save_path: Optional[str] = None
):
    """
    Plot aggregate rates and agent transaction patterns.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rate_history = abm_simulator.rate_history
    t = np.array(rate_history['t'])
    r_in = np.array(rate_history['r_in'])
    r_out = np.array(rate_history['r_out'])
    
    # Cash-in rate
    ax1 = axes[0, 0]
    ax1.plot(t, r_in, 'b-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cash-In Rate')
    ax1.set_title('Aggregate Cash-In Rate')
    ax1.grid(True, alpha=0.3)
    
    # Cash-out rate
    ax2 = axes[0, 1]
    ax2.plot(t, r_out, 'r-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cash-Out Rate')
    ax2.set_title('Aggregate Cash-Out Rate')
    ax2.grid(True, alpha=0.3)
    
    # Net flow rate
    ax3 = axes[1, 0]
    net_flow = r_in - r_out
    ax3.plot(t, net_flow, 'g-', linewidth=2)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.fill_between(t, 0, net_flow, where=(net_flow >= 0), alpha=0.3, color='green', label='Net Inflow')
    ax3.fill_between(t, 0, net_flow, where=(net_flow < 0), alpha=0.3, color='red', label='Net Outflow')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Net Flow Rate')
    ax3.set_title('Net Flow Rate (r_in - r_out)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Agent type distribution
    ax4 = axes[1, 1]
    from collections import Counter
    agent_types = [agent.agent_type.value for agent in abm_simulator.agents]
    type_counts = Counter(agent_types)
    
    ax4.bar(type_counts.keys(), type_counts.values(), color='skyblue', edgecolor='black')
    ax4.set_xlabel('Agent Type')
    ax4.set_ylabel('Count')
    ax4.set_title('Agent Population Distribution')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig
