"""
Revnet Archetype Simulations
=============================

Simulates the three main Revnet archetypes:
1. Token-Launchpad: Speculative, short issuance, then AMM trading
2. Stable-Commerce: Stablecoin-like, fixed price, loyalty/cashback
3. Periodic Fundraising: Staged rounds with periodic price increases

Based on the Revnet archetype specifications.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from revnet_simulator import (
    Stage, RevnetState, RevnetSimulator, 
    constant_rate, piecewise_rate, sinusoidal_rate
)
from analysis import AnalyticalSolver, CounterfactualAnalyzer
from visualization import plot_simulation_results

print("=" * 80)
print("REVNET ARCHETYPE SIMULATIONS")
print("=" * 80)
print()


# =============================================================================
# ARCHETYPE 1: TOKEN-LAUNCHPAD REVNET
# =============================================================================

def simulate_token_launchpad():
    """
    Archetype 1: Token-Launchpad Revnet
    
    Characteristics:
    - Short issuance period (~3 months)
    - Decreasing issuance (high initial, tapering off)
    - No price ceiling after TGE
    - Transitions to AMM trading phase
    - Revenue from trading fees
    """
    print("=" * 80)
    print("ARCHETYPE 1: TOKEN-LAUNCHPAD REVNET")
    print("=" * 80)
    print()
    print("Speculative, narrative-driven community token")
    print("Short issuance phase, then AMM trading dominates")
    print()
    
    # Parameters
    issuance_duration = 90  # 3 months (in days)
    
    # Stage 1: Active issuance (3 months)
    # Decreasing issuance: price increases rapidly to encourage early participation
    stage1 = Stage(
        t_start=0.0,
        P_issue_0=0.10,      # Start cheap: $0.10 per token
        gamma_cut=0.20,      # 20% price cut → gamma = 1.25 (25% increase per step)
        Delta_t=7.0,         # Price increases every week
        r_cashout=0.05,      # Low 5% cash-out tax (liquid market)
        sigma=0.15,          # 15% split for operations/team
    )
    
    # Stage 2: Post-TGE / AMM Phase
    # Price escapes (no more issuance, or minimal)
    # High issuance price discourages minting, market takes over
    stage2 = Stage(
        t_start=issuance_duration,
        P_issue_0=10.0,      # Very high price ($10) - discourages issuance
        gamma_cut=0.01,      # Minimal increases
        Delta_t=30.0,        # Monthly adjustments
        r_cashout=0.05,      # Keep same redemption tax
        sigma=0.15,
    )
    
    initial_state = RevnetState(B=1000.0, S=10000.0, t=0.0)
    
    # Activity rates: High during launch, decreases as AMM takes over
    def r_in_launchpad(t):
        """Decreasing cash-in rate during issuance, minimal after"""
        if t < issuance_duration:
            # High initial interest, tapering off
            # Exponential decay: starts at 500, decays to ~50
            return 500 * np.exp(-t / 30) + 50
        else:
            # Post-TGE: minimal issuance (most activity on AMM)
            return 5.0
    
    def r_out_launchpad(t):
        """Low redemption during issuance, moderate after launch"""
        if t < issuance_duration:
            # Some early sellers, but mostly holding for TGE
            return 20.0
        else:
            # Post-TGE: some profit-taking
            return 50.0 + 30 * np.sin(2 * np.pi * t / 60)  # Cyclical with ~2 month period
    
    sim = RevnetSimulator(
        stages=[stage1, stage2],
        initial_state=initial_state,
        phi_tot=0.02,  # 2% protocol fee
        seed=42
    )
    
    print("Configuration:")
    print(f"  Initial price: ${stage1.P_issue_0:.2f}")
    print(f"  Issuance duration: {issuance_duration} days")
    print(f"  Price increase: {(stage1.gamma - 1) * 100:.1f}% every {stage1.Delta_t} days")
    print(f"  Post-TGE price: ${stage2.P_issue_0:.2f} (discourages minting)")
    print(f"  Cash-out tax: {stage1.r_cashout * 100:.0f}%")
    print()
    
    print("Simulating 180 days (6 months)...")
    sim.simulate(t_end=180.0, r_in_func=r_in_launchpad, r_out_func=r_out_launchpad, dt=0.5)
    history = sim.get_history()
    
    # Analysis
    tge_idx = np.argmin(np.abs(history['t'] - issuance_duration))
    
    print(f"\nResults:")
    print(f"  At TGE (t={issuance_duration}):")
    print(f"    Treasury: ${history['B'][tge_idx]:,.2f}")
    print(f"    Supply: {history['S'][tge_idx]:,.2f} tokens")
    print(f"    Floor: ${history['P_floor'][tge_idx]:.4f}")
    print(f"    Issuance Price: ${history['P_issue'][tge_idx]:.4f}")
    print()
    print(f"  At End (t=180):")
    print(f"    Treasury: ${history['B'][-1]:,.2f}")
    print(f"    Supply: {history['S'][-1]:,.2f} tokens")
    print(f"    Floor: ${history['P_floor'][-1]:.4f}")
    print(f"    Floor Growth: {(history['P_floor'][-1] / history['P_floor'][0] - 1) * 100:+.1f}%")
    
    # Visualize
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    t = history['t']
    
    # Treasury
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, history['B'], 'b-', linewidth=2)
    ax1.axvline(issuance_duration, color='red', linestyle='--', alpha=0.7, label='TGE')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Treasury ($)')
    ax1.set_title('Treasury Growth')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Supply
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, history['S'], 'g-', linewidth=2)
    ax2.axvline(issuance_duration, color='red', linestyle='--', alpha=0.7, label='TGE')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Token Supply')
    ax2.set_title('Token Supply')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Floor vs Issuance Price
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(t, history['P_floor'], 'r-', linewidth=2, label='Floor Price')
    ax3.semilogy(t, history['P_issue'], 'k--', linewidth=2, alpha=0.7, label='Issuance Price')
    ax3.axvline(issuance_duration, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Price ($, log scale)')
    ax3.set_title('Floor vs Issuance Price')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Cash-in rate
    ax4 = fig.add_subplot(gs[1, 0])
    r_in_vals = [r_in_launchpad(ti) for ti in t]
    ax4.plot(t, r_in_vals, 'purple', linewidth=2)
    ax4.axvline(issuance_duration, color='red', linestyle='--', alpha=0.7)
    ax4.fill_between(t, 0, r_in_vals, alpha=0.3, color='purple')
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Cash-In Rate ($/day)')
    ax4.set_title('Issuance Activity (Decreasing)')
    ax4.grid(True, alpha=0.3)
    
    # Cash-out rate
    ax5 = fig.add_subplot(gs[1, 1])
    r_out_vals = [r_out_launchpad(ti) for ti in t]
    ax5.plot(t, r_out_vals, 'orange', linewidth=2)
    ax5.axvline(issuance_duration, color='red', linestyle='--', alpha=0.7)
    ax5.fill_between(t, 0, r_out_vals, alpha=0.3, color='orange')
    ax5.set_xlabel('Time (days)')
    ax5.set_ylabel('Cash-Out Rate (tokens/day)')
    ax5.set_title('Redemption Activity')
    ax5.grid(True, alpha=0.3)
    
    # Backing ratio
    ax6 = fig.add_subplot(gs[1, 2])
    backing = history['B'] / history['S']
    ax6.plot(t, backing, 'teal', linewidth=2)
    ax6.axvline(issuance_duration, color='red', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Time (days)')
    ax6.set_ylabel('Backing Ratio ($/token)')
    ax6.set_title('Treasury per Token')
    ax6.grid(True, alpha=0.3)
    
    # Spread (Issuance - Floor)
    ax7 = fig.add_subplot(gs[2, 0])
    spread = history['P_issue'] - history['P_floor']
    ax7.plot(t, spread, 'brown', linewidth=2)
    ax7.axvline(issuance_duration, color='red', linestyle='--', alpha=0.7)
    ax7.fill_between(t, 0, spread, alpha=0.3, color='brown')
    ax7.set_xlabel('Time (days)')
    ax7.set_ylabel('Price Spread ($)')
    ax7.set_title('Issuance Premium Over Floor')
    ax7.grid(True, alpha=0.3)
    
    # Phase diagram
    ax8 = fig.add_subplot(gs[2, 1])
    colors = np.where(t < issuance_duration, 'blue', 'green')
    ax8.scatter(history['S'], history['B'], c=colors, s=10, alpha=0.6)
    ax8.set_xlabel('Token Supply')
    ax8.set_ylabel('Treasury ($)')
    ax8.set_title('Phase Plot (Blue=Issuance, Green=AMM)')
    ax8.grid(True, alpha=0.3)
    
    # Market Cap vs Treasury
    ax9 = fig.add_subplot(gs[2, 2])
    market_cap = history['S'] * history['P_floor']
    ax9.plot(t, market_cap, 'darkgreen', linewidth=2, label='Market Cap (Floor)')
    ax9.plot(t, history['B'], 'darkblue', linewidth=2, label='Treasury')
    ax9.axvline(issuance_duration, color='red', linestyle='--', alpha=0.7)
    ax9.set_xlabel('Time (days)')
    ax9.set_ylabel('Value ($)')
    ax9.set_title('Market Cap vs Treasury')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    fig.suptitle('Archetype 1: Token-Launchpad Revnet\nSpeculative Launch → AMM Trading', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig('./results/archetype1_token_launchpad.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved to archetype1_token_launchpad.png")
    print()
    
    return history


# =============================================================================
# ARCHETYPE 2: STABLE-COMMERCE REVNET
# =============================================================================

def simulate_stable_commerce():
    """
    Archetype 2: Stable-Commerce Revnet
    
    Characteristics:
    - Fixed 1:1 issuance (1 token per 1 USDC)
    - Very high split (95-98%) - acts as cashback
    - No price ceiling/floor changes (stable)
    - Optional: small scheduled price increases
    - No AMM trading
    - Loyalty/reward system
    """
    print("=" * 80)
    print("ARCHETYPE 2: STABLE-COMMERCE REVNET")
    print("=" * 80)
    print()
    print("Stablecoin-like loyalty system for businesses")
    print("Fixed price, high cashback, no speculation")
    print()
    
    # Single stage: stable issuance
    # Optional: small increases every quarter to reward early customers
    stage = Stage(
        t_start=0.0,
        P_issue_0=1.0,       # Fixed: 1 token per 1 USDC
        gamma_cut=0.005,     # 0.5% increase per step (very small)
        Delta_t=90.0,        # Quarterly adjustments
        r_cashout=0.02,      # Very low 2% cash-out tax (easy redemption)
        sigma=0.97,          # 97% split - acts as cashback to customers
    )
    
    initial_state = RevnetState(B=1000.0, S=1000.0, t=0.0)
    
    # Activity rates: steady business activity
    def r_in_commerce(t):
        """Steady cash-in from business operations"""
        # Constant base + weekly cycle (busy weekends)
        base = 200.0
        weekly_cycle = 50 * np.sin(2 * np.pi * t / 7)
        # Slight growth over time as business grows
        growth = 1 + 0.001 * t
        return (base + weekly_cycle) * growth
    
    def r_out_commerce(t):
        """Steady redemptions - customers using cashback"""
        # Slightly less than cash-in (most hold for future purchases)
        # Similar weekly pattern
        base = 180.0
        weekly_cycle = 40 * np.sin(2 * np.pi * t / 7 + np.pi/4)  # Offset from cash-in
        growth = 1 + 0.001 * t
        return (base + weekly_cycle) * growth
    
    sim = RevnetSimulator(
        stages=[stage],
        initial_state=initial_state,
        phi_tot=0.0,  # No protocol fee for business use case
        seed=42
    )
    
    print("Configuration:")
    print(f"  Issuance price: ${stage.P_issue_0:.2f} (fixed)")
    print(f"  Price drift: {stage.gamma_cut * 100:.1f}% per quarter (optional)")
    print(f"  Split (cashback): {stage.sigma * 100:.0f}%")
    print(f"  Cash-out tax: {stage.r_cashout * 100:.0f}%")
    print(f"  Business use: Coffee shop loyalty program")
    print()
    
    print("Simulating 365 days (1 year)...")
    sim.simulate(t_end=365.0, r_in_func=r_in_commerce, r_out_func=r_out_commerce, dt=0.5)
    history = sim.get_history()
    
    # Analysis
    print(f"\nResults:")
    print(f"  Initial:")
    print(f"    Floor: ${history['P_floor'][0]:.4f}")
    print(f"    Issuance Price: ${history['P_issue'][0]:.4f}")
    print()
    print(f"  After 1 Year:")
    print(f"    Treasury: ${history['B'][-1]:,.2f}")
    print(f"    Supply: {history['S'][-1]:,.2f} tokens")
    print(f"    Floor: ${history['P_floor'][-1]:.4f}")
    print(f"    Issuance Price: ${history['P_issue'][-1]:.4f}")
    print(f"    Floor Stability: {np.std(history['P_floor']) / np.mean(history['P_floor']) * 100:.2f}% (lower is better)")
    
    # Visualize
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    t = history['t']
    
    # Treasury
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, history['B'], 'b-', linewidth=2)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Treasury ($)')
    ax1.set_title('Business Treasury')
    ax1.grid(True, alpha=0.3)
    
    # Supply
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, history['S'], 'g-', linewidth=2)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Token Supply')
    ax2.set_title('Loyalty Tokens in Circulation')
    ax2.grid(True, alpha=0.3)
    
    # Floor Price (should be very stable)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, history['P_floor'], 'r-', linewidth=2, label='Floor')
    ax3.plot(t, history['P_issue'], 'k--', linewidth=2, alpha=0.7, label='Issuance')
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Price ($)')
    ax3.set_title('Price Stability (Near $1)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0.95, 1.05])  # Zoom to show stability
    
    # Activity rates
    ax4 = fig.add_subplot(gs[1, 0])
    r_in_vals = [r_in_commerce(ti) for ti in t]
    ax4.plot(t, r_in_vals, 'purple', linewidth=2)
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Cash-In Rate ($/day)')
    ax4.set_title('Customer Purchases (Weekly Cycle)')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    r_out_vals = [r_out_commerce(ti) for ti in t]
    ax5.plot(t, r_out_vals, 'orange', linewidth=2)
    ax5.set_xlabel('Time (days)')
    ax5.set_ylabel('Redemption Rate (tokens/day)')
    ax5.set_title('Cashback Redemptions')
    ax5.grid(True, alpha=0.3)
    
    # Net flow
    ax6 = fig.add_subplot(gs[1, 2])
    net_flow = np.array(r_in_vals) - np.array(r_out_vals)
    ax6.plot(t, net_flow, 'teal', linewidth=2)
    ax6.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax6.fill_between(t, 0, net_flow, where=(net_flow >= 0), alpha=0.3, color='green', label='Net Accumulation')
    ax6.fill_between(t, 0, net_flow, where=(net_flow < 0), alpha=0.3, color='red', label='Net Redemption')
    ax6.set_xlabel('Time (days)')
    ax6.set_ylabel('Net Flow ($/day)')
    ax6.set_title('Net Daily Flow')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Backing ratio (should be ~1)
    ax7 = fig.add_subplot(gs[2, 0])
    backing = history['B'] / history['S']
    ax7.plot(t, backing, 'brown', linewidth=2)
    ax7.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Target: $1/token')
    ax7.set_xlabel('Time (days)')
    ax7.set_ylabel('Backing Ratio ($/token)')
    ax7.set_title('Treasury Backing per Token')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    ax7.set_ylim([0.95, 1.05])
    
    # Weekly pattern analysis (last 30 days)
    ax8 = fig.add_subplot(gs[2, 1])
    last_30_days = t >= (t[-1] - 30)
    t_30 = t[last_30_days] - t[last_30_days][0]  # Normalize to 0
    floor_30 = history['P_floor'][last_30_days]
    ax8.plot(t_30, floor_30, 'darkgreen', linewidth=2)
    ax8.set_xlabel('Days (last 30)')
    ax8.set_ylabel('Floor Price ($)')
    ax8.set_title('Price Stability (Last 30 Days)')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([0.98, 1.02])
    
    # Cumulative revenue (business perspective)
    ax9 = fig.add_subplot(gs[2, 2])
    # Business keeps (1-sigma) of each cash-in
    business_revenue = np.cumsum([r_in_commerce(ti) * 0.5 for ti in t]) * (1 - stage.sigma)
    customer_cashback = np.cumsum([r_in_commerce(ti) * 0.5 for ti in t]) * stage.sigma
    ax9.plot(t, business_revenue, 'darkblue', linewidth=2, label='Business Revenue')
    ax9.plot(t, customer_cashback, 'darkgreen', linewidth=2, label='Customer Cashback')
    ax9.set_xlabel('Time (days)')
    ax9.set_ylabel('Cumulative Value ($)')
    ax9.set_title('Revenue Split (97% Cashback)')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    fig.suptitle('Archetype 2: Stable-Commerce Revnet\nFixed Price, High Cashback, Loyalty System', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig('./results/archetype2_stable_commerce.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved to archetype2_stable_commerce.png")
    print()
    
    return history


# =============================================================================
# ARCHETYPE 3: PERIODIC FUNDRAISING ROUNDS
# =============================================================================

def simulate_periodic_fundraising():
    """
    Archetype 3: Periodic Fundraising Rounds
    
    Characteristics:
    - 30-90 day issuance periods
    - 30-50% price increase per period
    - Natural marketing/storytelling moments
    - Community engagement cycles
    - Between speculation and stability
    """
    print("=" * 80)
    print("ARCHETYPE 3: PERIODIC FUNDRAISING ROUNDS")
    print("=" * 80)
    print()
    print("Timed fundraising cycles with narrative marketing")
    print("30-90 day periods, 30-50% price increases")
    print()
    
    # Define multiple stages for periodic rounds
    # Each round: 60 days with 40% price increase
    n_rounds = 6
    period_length = 60  # days
    price_increase = 0.40  # 40% per round
    
    stages = []
    current_price = 1.0
    
    for i in range(n_rounds):
        stage = Stage(
            t_start=i * period_length,
            P_issue_0=current_price,
            gamma_cut=0.0,       # No intra-period price changes
            Delta_t=1000.0,      # Effectively constant within period
            r_cashout=0.15,      # Moderate 15% cash-out tax
            sigma=0.20,          # 20% split for operations
        )
        stages.append(stage)
        current_price *= (1 + price_increase)  # Next round starts 40% higher
    
    initial_state = RevnetState(B=5000.0, S=5000.0, t=0.0)
    
    # Activity rates: peaks at round transitions (marketing moments)
    def r_in_fundraising(t):
        """Cash-in peaks near round transitions (marketing windows)"""
        period_num = int(t / period_length)
        t_in_period = t % period_length
        
        # Base rate increases each round (growing community)
        base = 100 * (1.3 ** period_num)
        
        # Peak at start of round (new price announced) and end (FOMO before increase)
        if t_in_period < 10:
            # Start of round: high activity from announcement
            multiplier = 3.0 - 0.2 * t_in_period
        elif t_in_period > period_length - 15:
            # End of round: FOMO before price increase
            days_until_end = period_length - t_in_period
            multiplier = 1.0 + 1.5 * (1 - days_until_end / 15)
        else:
            # Mid-round: steady activity
            multiplier = 1.0
        
        return base * multiplier
    
    def r_out_fundraising(t):
        """Redemptions lower during peak interest, higher mid-round"""
        period_num = int(t / period_length)
        t_in_period = t % period_length
        
        base = 60 * (1.2 ** period_num)
        
        # Inverse of cash-in pattern
        if t_in_period < 10 or t_in_period > period_length - 15:
            # Low redemptions during marketing windows
            multiplier = 0.5
        else:
            # Higher mid-round (some profit-taking)
            multiplier = 1.2
        
        return base * multiplier
    
    sim = RevnetSimulator(
        stages=stages,
        initial_state=initial_state,
        phi_tot=0.02,  # 2% protocol fee
        seed=42
    )
    
    print("Configuration:")
    print(f"  Number of rounds: {n_rounds}")
    print(f"  Period length: {period_length} days")
    print(f"  Price increase per round: {price_increase * 100:.0f}%")
    print(f"  Initial price: ${stages[0].P_issue_0:.2f}")
    print(f"  Final round price: ${stages[-1].P_issue_0:.2f}")
    print(f"  Cash-out tax: {stages[0].r_cashout * 100:.0f}%")
    print()
    
    total_time = n_rounds * period_length
    print(f"Simulating {total_time} days ({n_rounds} rounds of {period_length} days)...")
    sim.simulate(t_end=total_time, r_in_func=r_in_fundraising, r_out_func=r_out_fundraising, dt=0.5)
    history = sim.get_history()
    
    # Analysis
    print(f"\nResults by Round:")
    for i in range(n_rounds):
        round_start = i * period_length
        round_end = (i + 1) * period_length
        round_idx_start = np.argmin(np.abs(history['t'] - round_start))
        round_idx_end = np.argmin(np.abs(history['t'] - round_end))
        
        treasury_growth = history['B'][round_idx_end] - history['B'][round_idx_start]
        supply_growth = history['S'][round_idx_end] - history['S'][round_idx_start]
        
        print(f"  Round {i+1} (Days {round_start}-{round_end}):")
        print(f"    Treasury raised: ${treasury_growth:,.2f}")
        print(f"    Tokens issued: {supply_growth:,.2f}")
        print(f"    Avg price: ${treasury_growth / supply_growth if supply_growth > 0 else 0:.3f}")
    
    print(f"\n  Overall:")
    print(f"    Total treasury: ${history['B'][-1]:,.2f}")
    print(f"    Total supply: {history['S'][-1]:,.2f}")
    print(f"    Final floor: ${history['P_floor'][-1]:.3f}")
    print(f"    Floor growth: {(history['P_floor'][-1] / history['P_floor'][0] - 1) * 100:+.1f}%")
    
    # Visualize
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    t = history['t']
    
    # Treasury with round markers
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, history['B'], 'b-', linewidth=2)
    for i in range(1, n_rounds):
        ax1.axvline(i * period_length, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Treasury ($)')
    ax1.set_title('Treasury Growth Across Rounds')
    ax1.grid(True, alpha=0.3)
    
    # Supply
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, history['S'], 'g-', linewidth=2)
    for i in range(1, n_rounds):
        ax2.axvline(i * period_length, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Token Supply')
    ax2.set_title('Token Supply Growth')
    ax2.grid(True, alpha=0.3)
    
    # Floor and Issuance Prices
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, history['P_floor'], 'r-', linewidth=2, label='Floor')
    ax3.plot(t, history['P_issue'], 'k-', linewidth=3, alpha=0.7, label='Issuance', drawstyle='steps-post')
    for i in range(1, n_rounds):
        ax3.axvline(i * period_length, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Price ($)')
    ax3.set_title('Price Evolution (40% Steps)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Cash-in rate with round cycles
    ax4 = fig.add_subplot(gs[1, 0])
    r_in_vals = [r_in_fundraising(ti) for ti in t]
    ax4.plot(t, r_in_vals, 'purple', linewidth=2)
    for i in range(1, n_rounds):
        ax4.axvline(i * period_length, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.fill_between(t, 0, r_in_vals, alpha=0.3, color='purple')
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Cash-In Rate ($/day)')
    ax4.set_title('Issuance Activity (Peaks at Marketing Windows)')
    ax4.grid(True, alpha=0.3)
    
    # Cash-out rate
    ax5 = fig.add_subplot(gs[1, 1])
    r_out_vals = [r_out_fundraising(ti) for ti in t]
    ax5.plot(t, r_out_vals, 'orange', linewidth=2)
    for i in range(1, n_rounds):
        ax5.axvline(i * period_length, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax5.fill_between(t, 0, r_out_vals, alpha=0.3, color='orange')
    ax5.set_xlabel('Time (days)')
    ax5.set_ylabel('Redemption Rate (tokens/day)')
    ax5.set_title('Redemption Activity')
    ax5.grid(True, alpha=0.3)
    
    # Backing ratio
    ax6 = fig.add_subplot(gs[1, 2])
    backing = history['B'] / history['S']
    ax6.plot(t, backing, 'teal', linewidth=2)
    for i in range(1, n_rounds):
        ax6.axvline(i * period_length, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax6.set_xlabel('Time (days)')
    ax6.set_ylabel('Backing Ratio ($/token)')
    ax6.set_title('Treasury per Token')
    ax6.grid(True, alpha=0.3)
    
    # Spread evolution
    ax7 = fig.add_subplot(gs[2, 0])
    spread = history['P_issue'] - history['P_floor']
    spread_pct = 100 * spread / history['P_issue']
    ax7.plot(t, spread_pct, 'brown', linewidth=2)
    for i in range(1, n_rounds):
        ax7.axvline(i * period_length, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax7.set_xlabel('Time (days)')
    ax7.set_ylabel('Spread (%)')
    ax7.set_title('Issuance Premium Over Floor (%)')
    ax7.grid(True, alpha=0.3)
    
    # Treasury growth per round
    ax8 = fig.add_subplot(gs[2, 1])
    round_growth = []
    round_labels = []
    for i in range(n_rounds):
        round_start = i * period_length
        round_end = (i + 1) * period_length
        idx_start = np.argmin(np.abs(history['t'] - round_start))
        idx_end = np.argmin(np.abs(history['t'] - round_end))
        growth = history['B'][idx_end] - history['B'][idx_start]
        round_growth.append(growth)
        round_labels.append(f"R{i+1}")
    
    bars = ax8.bar(round_labels, round_growth, color='darkblue', edgecolor='black', linewidth=1.5)
    ax8.set_xlabel('Round')
    ax8.set_ylabel('Treasury Raised ($)')
    ax8.set_title('Fundraising per Round')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, round_growth):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Phase plot
    ax9 = fig.add_subplot(gs[2, 2])
    # Color by round
    colors_phase = plt.cm.viridis(np.linspace(0, 1, n_rounds))
    for i in range(n_rounds):
        round_start = i * period_length
        round_end = (i + 1) * period_length
        mask = (t >= round_start) & (t < round_end)
        ax9.plot(history['S'][mask], history['B'][mask], 
                color=colors_phase[i], linewidth=2, label=f'R{i+1}')
    ax9.set_xlabel('Token Supply')
    ax9.set_ylabel('Treasury ($)')
    ax9.set_title('Phase Plot (Colored by Round)')
    ax9.grid(True, alpha=0.3)
    ax9.legend(loc='upper left', fontsize=8)
    
    # Single round detail (last round)
    ax10 = fig.add_subplot(gs[3, :])
    last_round_start = (n_rounds - 1) * period_length
    last_round_mask = t >= last_round_start
    t_last = t[last_round_mask] - last_round_start  # Normalize to round start
    
    # Plot multiple metrics
    ax10_twin1 = ax10.twinx()
    ax10_twin2 = ax10.twinx()
    ax10_twin2.spines['right'].set_position(('outward', 60))
    
    p1 = ax10.plot(t_last, [r_in_fundraising(ti) for ti in t[last_round_mask]], 
                   'purple', linewidth=2, label='Cash-In Rate')
    p2 = ax10_twin1.plot(t_last, history['P_floor'][last_round_mask], 
                         'red', linewidth=2, label='Floor Price')
    p3 = ax10_twin2.plot(t_last, history['B'][last_round_mask] - history['B'][last_round_mask][0], 
                         'blue', linewidth=2, label='Treasury Growth')
    
    ax10.set_xlabel('Days in Round')
    ax10.set_ylabel('Cash-In Rate ($/day)', color='purple')
    ax10_twin1.set_ylabel('Floor Price ($)', color='red')
    ax10_twin2.set_ylabel('Treasury Growth ($)', color='blue')
    ax10.set_title(f'Round {n_rounds} Detail: Marketing Peaks at Start and End')
    ax10.grid(True, alpha=0.3)
    
    # Add shaded regions for marketing windows
    ax10.axvspan(0, 10, alpha=0.2, color='green', label='Launch Marketing')
    ax10.axvspan(period_length - 15, period_length, alpha=0.2, color='orange', label='FOMO Window')
    
    # Combine legends
    lns = p1 + p2 + p3
    labs = [l.get_label() for l in lns]
    ax10.legend(lns, labs, loc='upper left')
    
    fig.suptitle('Archetype 3: Periodic Fundraising Rounds\nTimed Cycles with Marketing Moments', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig('./results/archetype3_periodic_fundraising.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved to archetype3_periodic_fundraising.png")
    print()
    
    return history


# =============================================================================
# COMPARATIVE ANALYSIS
# =============================================================================

def compare_archetypes():
    """
    Compare all three archetypes side-by-side
    """
    print("=" * 80)
    print("COMPARATIVE ANALYSIS: ALL THREE ARCHETYPES")
    print("=" * 80)
    print()
    
    print("Running all three archetypes for comparison...")
    print()
    
    # Run simplified versions for comparison (shorter time horizon)
    t_compare = 180  # days
    
    # Archetype 1: Token Launchpad (simplified)
    stage1_simple = Stage(t_start=0.0, P_issue_0=0.10, gamma_cut=0.20, Delta_t=7.0, 
                          r_cashout=0.05, sigma=0.15)
    sim1 = RevnetSimulator([stage1_simple], RevnetState(1000, 10000, 0), 0.02, 42)
    sim1.simulate(t_compare, 
                  lambda t: 500 * np.exp(-t/30) + 50 if t < 90 else 5,
                  lambda t: 20 if t < 90 else 50, 
                  0.5)
    hist1 = sim1.get_history()
    
    # Archetype 2: Stable Commerce (simplified)
    stage2_simple = Stage(t_start=0.0, P_issue_0=1.0, gamma_cut=0.005, Delta_t=90.0,
                          r_cashout=0.02, sigma=0.97)
    sim2 = RevnetSimulator([stage2_simple], RevnetState(1000, 1000, 0), 0.0, 42)
    sim2.simulate(t_compare,
                  lambda t: 200 + 50 * np.sin(2*np.pi*t/7),
                  lambda t: 180 + 40 * np.sin(2*np.pi*t/7 + np.pi/4),
                  0.5)
    hist2 = sim2.get_history()
    
    # Archetype 3: Periodic Fundraising (3 rounds)
    stages3_simple = [
        Stage(t_start=0, P_issue_0=1.0, gamma_cut=0.0, Delta_t=1000, r_cashout=0.15, sigma=0.20),
        Stage(t_start=60, P_issue_0=1.4, gamma_cut=0.0, Delta_t=1000, r_cashout=0.15, sigma=0.20),
        Stage(t_start=120, P_issue_0=1.96, gamma_cut=0.0, Delta_t=1000, r_cashout=0.15, sigma=0.20),
    ]
    sim3 = RevnetSimulator(stages3_simple, RevnetState(5000, 5000, 0), 0.02, 42)
    def r_in_3(t):
        period = int(t / 60)
        t_in = t % 60
        base = 100 * (1.3 ** period)
        mult = 3.0 - 0.2*t_in if t_in < 10 else (1 + 1.5*(1-(60-t_in)/15) if t_in > 45 else 1.0)
        return base * mult
    sim3.simulate(t_compare, r_in_3, lambda t: 60 * (1.2 ** int(t/60)), 0.5)
    hist3 = sim3.get_history()
    
    # Comparative plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    archetypes = [
        ('Token-Launchpad', hist1, 'blue'),
        ('Stable-Commerce', hist2, 'green'),
        ('Periodic Fundraising', hist3, 'red')
    ]
    
    # Floor price comparison
    ax = axes[0, 0]
    for name, hist, color in archetypes:
        ax.plot(hist['t'], hist['P_floor'], linewidth=2, label=name, color=color)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Floor Price ($)')
    ax.set_title('Floor Price Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Treasury growth
    ax = axes[0, 1]
    for name, hist, color in archetypes:
        treasury_growth = (hist['B'] - hist['B'][0]) / hist['B'][0] * 100
        ax.plot(hist['t'], treasury_growth, linewidth=2, label=name, color=color)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Treasury Growth (%)')
    ax.set_title('Treasury Growth Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Supply growth
    ax = axes[0, 2]
    for name, hist, color in archetypes:
        supply_growth = (hist['S'] - hist['S'][0]) / hist['S'][0] * 100
        ax.plot(hist['t'], supply_growth, linewidth=2, label=name, color=color)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Supply Growth (%)')
    ax.set_title('Token Supply Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Backing ratio
    ax = axes[1, 0]
    for name, hist, color in archetypes:
        backing = hist['B'] / hist['S']
        ax.plot(hist['t'], backing, linewidth=2, label=name, color=color)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Backing Ratio ($/token)')
    ax.set_title('Treasury Backing per Token')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spread (Issuance - Floor) %
    ax = axes[1, 1]
    for name, hist, color in archetypes:
        spread_pct = 100 * (hist['P_issue'] - hist['P_floor']) / hist['P_issue']
        ax.plot(hist['t'], spread_pct, linewidth=2, label=name, color=color)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Spread (%)')
    ax.set_title('Issuance Premium Over Floor (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary metrics table
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = []
    headers = ['Archetype', 'Final Floor', 'Treasury', 'Volatility']
    
    for name, hist, color in archetypes:
        final_floor = hist['P_floor'][-1]
        final_treasury = hist['B'][-1]
        floor_volatility = np.std(np.diff(hist['P_floor'])) / np.mean(hist['P_floor']) * 100
        
        table_data.append([
            name,
            f"${final_floor:.3f}",
            f"${final_treasury:,.0f}",
            f"{floor_volatility:.2f}%"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.2, 1, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by archetype
    for i, (name, hist, color) in enumerate(archetypes, 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.2)
    
    ax.text(0.5, 0.05, 'Summary Metrics (180 days)', 
            ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    
    fig.suptitle('Comparative Analysis: Three Revnet Archetypes', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/archetype_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Comparative analysis complete")
    print("✓ Plot saved to archetype_comparison.png")
    print()
    
    # Print summary
    print("Summary Comparison:")
    print("-" * 80)
    for name, hist, color in archetypes:
        print(f"\n{name}:")
        print(f"  Final Floor: ${hist['P_floor'][-1]:.3f}")
        print(f"  Final Treasury: ${hist['B'][-1]:,.2f}")
        print(f"  Floor Growth: {(hist['P_floor'][-1]/hist['P_floor'][0] - 1)*100:+.1f}%")
        print(f"  Supply Growth: {(hist['S'][-1]/hist['S'][0] - 1)*100:+.1f}%")
        floor_vol = np.std(np.diff(hist['P_floor'])) / np.mean(hist['P_floor']) * 100
        print(f"  Price Volatility: {floor_vol:.2f}% (lower is more stable)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all archetype simulations"""
    
    # Run individual simulations
    hist1 = simulate_token_launchpad()
    hist2 = simulate_stable_commerce()
    hist3 = simulate_periodic_fundraising()
    
    # Comparative analysis
    compare_archetypes()
    
    print()
    print("=" * 80)
    print("ALL ARCHETYPE SIMULATIONS COMPLETE")
    print("=" * 80)
    print()
    print("Generated plots:")
    print("  1. archetype1_token_launchpad.png")
    print("  2. archetype2_stable_commerce.png")
    print("  3. archetype3_periodic_fundraising.png")
    print("  4. archetype_comparison.png")
    print()
    print("Key Insights:")
    print()
    print("Archetype 1 (Token-Launchpad):")
    print("  - Rapid early growth from high initial demand")
    print("  - Price increases encourage early participation")
    print("  - Transitions to AMM phase after ~3 months")
    print("  - Suitable for speculative, meme-driven communities")
    print()
    print("Archetype 2 (Stable-Commerce):")
    print("  - Extremely stable price (~$1.00 ± 1%)")
    print("  - High cashback (97%) maximizes customer value")
    print("  - Weekly cycles reflect real business patterns")
    print("  - Perfect for loyalty programs and merchant adoption")
    print()
    print("Archetype 3 (Periodic Fundraising):")
    print("  - Clear marketing moments at round transitions")
    print("  - 40% price increases create fundraising urgency")
    print("  - Balances speculation and stability")
    print("  - Natural cadence for community updates")
    print()


if __name__ == "__main__":
    main()