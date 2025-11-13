"""
Revnet Agent-Based Model Dashboard
===================================

Professional Streamlit dashboard for exploring Revnet dynamics with
agent-based modeling and interactive Plotly visualizations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Tuple
import json

# Import simulator modules
from revnet_simulator import Stage, RevnetState, RevnetSimulator
from agent_based_model import (
    Agent, AgentType, AgentBasedRevnetSimulator,
    create_agent_population
)

# Page configuration
st.set_page_config(
    page_title="Revnet ABM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'history' not in st.session_state:
    st.session_state.history = None
if 'abm_sim' not in st.session_state:
    st.session_state.abm_sim = None
if 'archetype_presets' not in st.session_state:
    st.session_state.archetype_presets = {
        "Custom": None,
        "Token Launchpad": {
            "P_issue_0": 0.10,
            "gamma_cut": 0.20,
            "Delta_t": 7.0,
            "r_cashout": 0.05,
            "sigma": 0.15,
            "duration": 180,
            "agent_dist": {
                AgentType.RANDOM: 0.20,
                AgentType.PRICE_SENSITIVE: 0.25,
                AgentType.FLOOR_TRADER: 0.15,
                AgentType.MOMENTUM: 0.15,
                AgentType.HODLER: 0.15,
                AgentType.ARBITRAGEUR: 0.10
            }
        },
        "Stable Commerce": {
            "P_issue_0": 1.0,
            "gamma_cut": 0.005,
            "Delta_t": 90.0,
            "r_cashout": 0.02,
            "sigma": 0.97,
            "duration": 365,
            "agent_dist": {
                AgentType.RANDOM: 0.40,
                AgentType.PRICE_SENSITIVE: 0.30,
                AgentType.HODLER: 0.30,
            }
        },
        "Periodic Fundraising": {
            "P_issue_0": 1.0,
            "gamma_cut": 0.0,
            "Delta_t": 1000.0,
            "r_cashout": 0.15,
            "sigma": 0.20,
            "duration": 360,
            "agent_dist": {
                AgentType.RANDOM: 0.15,
                AgentType.PRICE_SENSITIVE: 0.30,
                AgentType.MOMENTUM: 0.25,
                AgentType.HODLER: 0.20,
                AgentType.ARBITRAGEUR: 0.10
            }
        }
    }


def main():
    """Main dashboard application"""

    # Header
    st.markdown('<div class="main-header">🚀 Revnet Agent-Based Model Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore autonomous tokenized financial structures with heterogeneous agents</div>',
                unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Archetype selection
        st.subheader("1. Select Archetype")
        archetype = st.selectbox(
            "Choose a preset or customize:",
            list(st.session_state.archetype_presets.keys()),
            help="Start with a predefined archetype or build your own"
        )

        # Load preset if selected
        if archetype != "Custom" and st.session_state.archetype_presets[archetype]:
            preset = st.session_state.archetype_presets[archetype]
            default_price = preset["P_issue_0"]
            default_gamma = preset["gamma_cut"]
            default_delta = preset["Delta_t"]
            default_cashout = preset["r_cashout"]
            default_sigma = preset["sigma"]
            default_duration = preset["duration"]
            default_agent_dist = preset["agent_dist"]
        else:
            default_price = 1.0
            default_gamma = 0.05
            default_delta = 30.0
            default_cashout = 0.10
            default_sigma = 0.50
            default_duration = 180
            default_agent_dist = {at: 1.0/6 for at in AgentType}

        st.markdown("---")

        # Stage parameters
        st.subheader("2. Stage Parameters")

        col1, col2 = st.columns(2)
        with col1:
            P_issue_0 = st.number_input(
                "Initial Price ($)",
                min_value=0.01,
                max_value=100.0,
                value=default_price,
                step=0.01,
                help="Starting issuance price per token"
            )

            gamma_cut = st.slider(
                "Price Cut (%)",
                min_value=0.0,
                max_value=50.0,
                value=default_gamma * 100,
                step=0.5,
                help="Price decay parameter: gamma = 1/(1-cut)"
            ) / 100

            r_cashout = st.slider(
                "Cash-out Tax (%)",
                min_value=0.0,
                max_value=50.0,
                value=default_cashout * 100,
                step=1.0,
                help="Tax on redemptions"
            ) / 100

        with col2:
            Delta_t = st.number_input(
                "Price Step (days)",
                min_value=1.0,
                max_value=365.0,
                value=default_delta,
                step=1.0,
                help="Days between price adjustments"
            )

            sigma = st.slider(
                "Split Ratio (%)",
                min_value=0.0,
                max_value=99.0,
                value=default_sigma * 100,
                step=1.0,
                help="Per-mint revenue split"
            ) / 100

            phi_tot = st.slider(
                "Protocol Fee (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Protocol token fee"
            ) / 100

        st.markdown("---")

        # Initial conditions
        st.subheader("3. Initial State")
        col1, col2 = st.columns(2)
        with col1:
            initial_treasury = st.number_input(
                "Initial Treasury ($)",
                min_value=100.0,
                max_value=1000000.0,
                value=5000.0,
                step=100.0
            )
        with col2:
            initial_supply = st.number_input(
                "Initial Supply",
                min_value=100.0,
                max_value=1000000.0,
                value=5000.0,
                step=100.0
            )

        st.markdown("---")

        # Agent configuration
        st.subheader("4. Agent Population")

        n_agents = st.slider(
            "Number of Agents",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Total agents in simulation"
        )

        st.write("**Agent Type Distribution**")
        agent_dist = {}
        remaining = 100.0

        agent_types = list(AgentType)
        for i, agent_type in enumerate(agent_types[:-1]):
            default_val = default_agent_dist.get(agent_type, 0.0) * 100
            max_val = min(remaining, 100.0)

            val = st.slider(
                f"{agent_type.value.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=max_val,
                value=min(default_val, max_val),
                step=5.0,
                key=f"agent_{agent_type.value}"
            ) / 100
            agent_dist[agent_type] = val
            remaining -= val * 100

        # Last agent type gets remainder
        agent_dist[agent_types[-1]] = max(0.0, remaining / 100)
        st.info(f"{agent_types[-1].value.replace('_', ' ').title()}: {remaining:.1f}% (auto)")

        # Normalize to ensure sum = 1
        total = sum(agent_dist.values())
        if total > 0:
            agent_dist = {k: v/total for k, v in agent_dist.items()}

        st.markdown("---")

        # Simulation parameters
        st.subheader("5. Simulation")

        duration = st.slider(
            "Duration (days)",
            min_value=30,
            max_value=730,
            value=default_duration,
            step=30,
            help="Simulation time horizon"
        )

        dt = st.select_slider(
            "Time Step",
            options=[0.01, 0.05, 0.1, 0.5, 1.0],
            value=0.1,
            help="Simulation resolution (smaller = more accurate but slower)"
        )

        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="For reproducibility"
        )

        st.markdown("---")

        # Run simulation button
        run_button = st.button(
            "🚀 Run Simulation",
            type="primary",
            use_container_width=True
        )

    # Main content area
    if run_button:
        with st.spinner("Running agent-based simulation... This may take a moment."):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create stage
            status_text.text("Creating stage configuration...")
            progress_bar.progress(10)

            stage = Stage(
                t_start=0.0,
                P_issue_0=P_issue_0,
                gamma_cut=gamma_cut,
                Delta_t=Delta_t,
                r_cashout=r_cashout,
                sigma=sigma
            )

            # Create initial state
            initial_state = RevnetState(
                B=initial_treasury,
                S=initial_supply,
                t=0.0
            )

            # Create agent population
            status_text.text(f"Creating {n_agents} agents with heterogeneous strategies...")
            progress_bar.progress(30)

            agents = create_agent_population(
                n_agents=n_agents,
                agent_type_distribution=agent_dist,
                seed=seed
            )

            # Create ABM simulator
            status_text.text("Initializing agent-based simulator...")
            progress_bar.progress(40)

            abm_sim = AgentBasedRevnetSimulator(
                stages=[stage],
                initial_state=initial_state,
                agents=agents,
                phi_tot=phi_tot,
                seed=seed
            )

            # Store initial state before simulation
            st.session_state.initial_B = initial_treasury
            st.session_state.initial_S = initial_supply

            # Run simulation
            status_text.text(f"Simulating {duration} days of agent interactions...")
            progress_bar.progress(50)

            start_time = time.time()
            abm_sim.simulate(t_end=duration, dt=dt)
            elapsed_time = time.time() - start_time

            progress_bar.progress(90)
            status_text.text("Processing results...")

            # Store results
            st.session_state.abm_sim = abm_sim
            st.session_state.simulation_run = True

            progress_bar.progress(100)
            status_text.text(f"✅ Simulation complete! ({elapsed_time:.2f}s)")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

    # Display results if simulation has been run
    if st.session_state.simulation_run and st.session_state.abm_sim:
        display_results(st.session_state.abm_sim, archetype)


def display_results(abm_sim: AgentBasedRevnetSimulator, archetype: str):
    """Display simulation results with interactive visualizations"""

    # Get state history
    state = abm_sim.state
    history = abm_sim.rate_history
    transactions = abm_sim.transactions

    # Convert to arrays
    t_array = np.array(history['t'])
    r_in_array = np.array(history['r_in'])
    r_out_array = np.array(history['r_out'])

    # Get initial state stored before simulation
    initial_B = st.session_state.get('initial_B', state.B)
    initial_S = st.session_state.get('initial_S', state.S)

    # Compute derived metrics
    P_floor_array = []
    P_issue_array = []

    # Build time series
    for i, t in enumerate(t_array):
        stage = abm_sim.get_current_stage(t)
        P_floor_array.append(state.marginal_floor(stage.r_cashout, abm_sim.phi_tot))
        P_issue_array.append(stage.issuance_price(t))

    # Summary metrics
    st.subheader("📊 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Treasury",
            f"${state.B:,.2f}",
            f"+{((state.B / initial_B - 1) * 100):.1f}%" if initial_B > 0 else "N/A"
        )

    with col2:
        st.metric(
            "Final Supply",
            f"{state.S:,.0f}",
            f"+{((state.S / initial_S - 1) * 100):.1f}%" if initial_S > 0 else "N/A"
        )

    with col3:
        final_floor = P_floor_array[-1]
        initial_floor = P_floor_array[0]
        st.metric(
            "Floor Price",
            f"${final_floor:.4f}",
            f"+{((final_floor / initial_floor - 1) * 100):.1f}%"
        )

    with col4:
        st.metric(
            "Total Transactions",
            f"{len(transactions):,}",
            f"{len(transactions) / (t_array[-1] if len(t_array) > 0 else 1):.1f}/day"
        )

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Market Dynamics",
        "👥 Agent Activity",
        "💰 Transactions",
        "📊 Analytics"
    ])

    with tab1:
        display_market_dynamics(t_array, r_in_array, r_out_array, P_floor_array, P_issue_array, state)

    with tab2:
        display_agent_activity(abm_sim, transactions)

    with tab3:
        display_transactions(transactions, t_array[-1] if len(t_array) > 0 else 180)

    with tab4:
        display_analytics(abm_sim, t_array, P_floor_array, archetype)


def display_market_dynamics(t_array, r_in_array, r_out_array, P_floor_array, P_issue_array, state):
    """Display market dynamics visualizations"""

    st.subheader("Price Evolution & Trading Activity")

    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Floor vs Issuance Price",
            "Cash-In Activity",
            "Cash-Out Activity",
            "Price Spread"
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )

    # Floor vs Issuance Price
    fig.add_trace(
        go.Scatter(
            x=t_array, y=P_floor_array,
            name="Floor Price",
            line=dict(color='#10b981', width=3),
            hovertemplate='Day %{x:.1f}<br>Floor: $%{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=t_array, y=P_issue_array,
            name="Issuance Price",
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='Day %{x:.1f}<br>Issuance: $%{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Cash-in rate
    fig.add_trace(
        go.Scatter(
            x=t_array, y=r_in_array,
            name="Cash-In Rate",
            fill='tozeroy',
            line=dict(color='#8b5cf6', width=2),
            hovertemplate='Day %{x:.1f}<br>Rate: $%{y:.2f}/day<extra></extra>'
        ),
        row=1, col=2
    )

    # Cash-out rate
    fig.add_trace(
        go.Scatter(
            x=t_array, y=r_out_array,
            name="Cash-Out Rate",
            fill='tozeroy',
            line=dict(color='#f59e0b', width=2),
            hovertemplate='Day %{x:.1f}<br>Rate: %{y:.2f} tokens/day<extra></extra>'
        ),
        row=2, col=1
    )

    # Price spread
    spread = np.array(P_issue_array) - np.array(P_floor_array)
    fig.add_trace(
        go.Scatter(
            x=t_array, y=spread,
            name="Price Spread",
            fill='tozeroy',
            line=dict(color='#06b6d4', width=2),
            hovertemplate='Day %{x:.1f}<br>Spread: $%{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Rate ($/day)", row=1, col=2)
    fig.update_yaxes(title_text="Rate (tokens/day)", row=2, col=1)
    fig.update_yaxes(title_text="Spread ($)", row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


def display_agent_activity(abm_sim: AgentBasedRevnetSimulator, transactions: List[Dict]):
    """Display agent-specific activity and holdings"""

    st.subheader("Agent Portfolio Distribution")

    # Aggregate agent holdings
    agent_data = []
    for agent in abm_sim.agents:
        agent_data.append({
            'Agent ID': agent.agent_id,
            'Type': agent.agent_type.value.replace('_', ' ').title(),
            'Token Balance': agent.token_balance,
            'Base Balance': agent.base_balance,
            'Total Value ($)': agent.base_balance + agent.token_balance * abm_sim.state.marginal_floor(
                abm_sim.get_current_stage(abm_sim.state.t).r_cashout,
                abm_sim.phi_tot
            )
        })

    df_agents = pd.DataFrame(agent_data)

    # Agent type distribution
    col1, col2 = st.columns(2)

    with col1:
        # Holdings by agent type
        type_holdings = df_agents.groupby('Type').agg({
            'Token Balance': 'sum',
            'Base Balance': 'sum',
            'Total Value ($)': 'sum'
        }).reset_index()

        fig = go.Figure(data=[
            go.Bar(
                x=type_holdings['Type'],
                y=type_holdings['Token Balance'],
                name='Token Holdings',
                marker_color='#8b5cf6'
            )
        ])
        fig.update_layout(
            title="Token Holdings by Agent Type",
            xaxis_title="Agent Type",
            yaxis_title="Total Tokens",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Portfolio value distribution
        fig = px.pie(
            type_holdings,
            values='Total Value ($)',
            names='Type',
            title='Portfolio Value Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Top agents table
    st.subheader("Top 10 Agents by Total Value")
    top_agents = df_agents.nlargest(10, 'Total Value ($)')

    st.dataframe(
        top_agents.style.format({
            'Token Balance': '{:.2f}',
            'Base Balance': '{:.2f}',
            'Total Value ($)': '${:,.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )


def display_transactions(transactions: List[Dict], max_time: float):
    """Display transaction history and analysis"""

    st.subheader("Transaction History")

    if not transactions:
        st.info("No transactions recorded in this simulation.")
        return

    df_tx = pd.DataFrame(transactions)

    # Transaction timeline
    fig = go.Figure()

    # Cash-in transactions
    cash_in = df_tx[df_tx['action'] == 'cash_in']
    if len(cash_in) > 0:
        fig.add_trace(go.Scatter(
            x=cash_in['t'],
            y=cash_in['amount'],
            mode='markers',
            name='Cash In',
            marker=dict(color='#10b981', size=8, opacity=0.6),
            hovertemplate='Day %{x:.2f}<br>Amount: $%{y:.2f}<extra></extra>'
        ))

    # Redemption transactions
    redeem = df_tx[df_tx['action'] == 'redeem']
    if len(redeem) > 0:
        fig.add_trace(go.Scatter(
            x=redeem['t'],
            y=redeem['amount'],
            mode='markers',
            name='Redemption',
            marker=dict(color='#ef4444', size=8, opacity=0.6),
            hovertemplate='Day %{x:.2f}<br>Amount: $%{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title="Transaction Timeline",
        xaxis_title="Time (days)",
        yaxis_title="Transaction Amount ($)",
        template='plotly_white',
        height=400,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Transaction statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Cash-Ins", f"{len(cash_in):,}")
        st.metric("Total Volume", f"${cash_in['amount'].sum():,.2f}" if len(cash_in) > 0 else "$0")

    with col2:
        st.metric("Total Redemptions", f"{len(redeem):,}")
        st.metric("Total Volume", f"${redeem['amount'].sum():,.2f}" if len(redeem) > 0 else "$0")

    with col3:
        net_flow = (cash_in['amount'].sum() if len(cash_in) > 0 else 0) - \
                   (redeem['amount'].sum() if len(redeem) > 0 else 0)
        st.metric("Net Flow", f"${net_flow:,.2f}")

    # Recent transactions table
    st.subheader("Recent Transactions (Last 50)")
    recent = df_tx.tail(50).sort_values('t', ascending=False)

    st.dataframe(
        recent[['t', 'agent_id', 'action', 'amount', 'price']].style.format({
            't': '{:.2f}',
            'amount': '${:.2f}',
            'price': '${:.4f}'
        }),
        use_container_width=True,
        hide_index=True
    )


def display_analytics(abm_sim: AgentBasedRevnetSimulator, t_array, P_floor_array, archetype: str):
    """Display advanced analytics and insights"""

    st.subheader("Advanced Analytics")

    # Performance metrics
    col1, col2 = st.columns(2)

    with col1:
        # Volatility analysis
        if len(P_floor_array) > 1:
            returns = np.diff(P_floor_array) / P_floor_array[:-1]
            volatility = np.std(returns) * 100

            st.metric(
                "Floor Price Volatility",
                f"{volatility:.2f}%",
                help="Standard deviation of price returns"
            )

        # Backing ratio
        backing = abm_sim.state.B / abm_sim.state.S
        st.metric(
            "Current Backing Ratio",
            f"${backing:.4f}",
            help="Treasury per token"
        )

    with col2:
        # Agent diversity
        agent_types = set(agent.agent_type for agent in abm_sim.agents)
        st.metric(
            "Active Agent Types",
            f"{len(agent_types)}/{len(AgentType)}"
        )

        # Activity rate
        if len(abm_sim.transactions) > 0 and len(t_array) > 0:
            activity_rate = len(abm_sim.transactions) / t_array[-1]
            st.metric(
                "Avg Transactions/Day",
                f"{activity_rate:.1f}"
            )

    # Export section
    st.markdown("---")
    st.subheader("📥 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export configuration
        config = {
            "archetype": archetype,
            "stage": {
                "P_issue_0": abm_sim.stages[0].P_issue_0,
                "gamma_cut": abm_sim.stages[0].gamma_cut,
                "Delta_t": abm_sim.stages[0].Delta_t,
                "r_cashout": abm_sim.stages[0].r_cashout,
                "sigma": abm_sim.stages[0].sigma
            },
            "initial_state": {
                "B": st.session_state.initial_B,
                "S": st.session_state.initial_S
            },
            "final_state": {
                "B": abm_sim.state.B,
                "S": abm_sim.state.S
            },
            "n_agents": len(abm_sim.agents),
            "duration": t_array[-1] if len(t_array) > 0 else 0,
            "total_transactions": len(abm_sim.transactions)
        }

        st.download_button(
            "Download Config (JSON)",
            data=json.dumps(config, indent=2),
            file_name=f"revnet_config_{archetype.lower().replace(' ', '_')}.json",
            mime="application/json"
        )

    with col2:
        # Export transactions
        if abm_sim.transactions:
            df_tx = pd.DataFrame(abm_sim.transactions)
            st.download_button(
                "Download Transactions (CSV)",
                data=df_tx.to_csv(index=False),
                file_name="revnet_transactions.csv",
                mime="text/csv"
            )

    with col3:
        # Export time series
        df_ts = pd.DataFrame({
            'time': t_array,
            'r_in': abm_sim.rate_history['r_in'],
            'r_out': abm_sim.rate_history['r_out'],
        })
        st.download_button(
            "Download Time Series (CSV)",
            data=df_ts.to_csv(index=False),
            file_name="revnet_timeseries.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
