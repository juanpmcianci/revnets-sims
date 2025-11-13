"""
Agent-Based Modeling for Revnet
================================

Implements agent-based simulation with heterogeneous agents making
issuance and redemption decisions based on various strategies.
"""

import numpy as np
from typing import List, Optional, Callable, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from revnet_simulator import RevnetState, Stage


class AgentType(Enum):
    """Types of agents with different behavioral strategies"""
    RANDOM = "random"
    PRICE_SENSITIVE = "price_sensitive"
    FLOOR_TRADER = "floor_trader"
    MOMENTUM = "momentum"
    HODLER = "hodler"
    ARBITRAGEUR = "arbitrageur"


@dataclass
class Agent:
    """
    Individual agent with strategy and holdings.
    
    Attributes:
    -----------
    agent_id : int
        Unique identifier
    agent_type : AgentType
        Behavioral strategy type
    token_balance : float
        Current token holdings
    base_balance : float
        Current base asset holdings
    parameters : Dict
        Strategy-specific parameters
    """
    agent_id: int
    agent_type: AgentType
    token_balance: float = 0.0
    base_balance: float = 100.0  # Initial endowment
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class AgentBasedRevnetSimulator:
    """
    Agent-based model for Revnet with heterogeneous agents.
    
    Converts discrete agent actions into aggregate rate functions
    for the continuous-time ODE system.
    """
    
    def __init__(
        self,
        stages: List[Stage],
        initial_state: RevnetState,
        agents: List[Agent],
        phi_tot: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize ABM simulator.
        
        Parameters:
        -----------
        stages : List[Stage]
            Revnet stages
        initial_state : RevnetState
            Initial system state
        agents : List[Agent]
            List of agents
        phi_tot : float
            Protocol token fee
        seed : Optional[int]
            Random seed
        """
        self.stages = sorted(stages, key=lambda s: s.t_start)
        self.state = initial_state
        self.agents = agents
        self.phi_tot = phi_tot
        self.rng = np.random.RandomState(seed)
        
        # Transaction history
        self.transactions = []
        
        # Time series of aggregate rates (to be computed)
        self.rate_history = {
            't': [],
            'r_in': [],
            'r_out': [],
        }
    
    def get_current_stage(self, t: float) -> Stage:
        """Get active stage at time t"""
        for i in range(len(self.stages) - 1, -1, -1):
            if t >= self.stages[i].t_start:
                return self.stages[i]
        raise ValueError(f"No stage found for time {t}")
    
    def compute_agent_action(
        self,
        agent: Agent,
        t: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Compute agent's action for time interval [t, t+dt).
        
        Returns:
        --------
        (cash_in_amount, token_redeem_amount) : Tuple[float, float]
            Amount of base asset to pay in and tokens to redeem
        """
        stage = self.get_current_stage(t)
        P_issue = stage.issuance_price(t)
        P_floor = self.state.marginal_floor(stage.r_cashout, self.phi_tot)
        
        if agent.agent_type == AgentType.RANDOM:
            return self._random_agent_action(agent, dt)
        
        elif agent.agent_type == AgentType.PRICE_SENSITIVE:
            return self._price_sensitive_action(agent, t, dt, P_issue, P_floor)
        
        elif agent.agent_type == AgentType.FLOOR_TRADER:
            return self._floor_trader_action(agent, t, dt, P_issue, P_floor)
        
        elif agent.agent_type == AgentType.MOMENTUM:
            return self._momentum_action(agent, t, dt, P_floor)
        
        elif agent.agent_type == AgentType.HODLER:
            return self._hodler_action(agent, t, dt, P_issue)
        
        elif agent.agent_type == AgentType.ARBITRAGEUR:
            return self._arbitrageur_action(agent, t, dt, P_issue, P_floor)
        
        else:
            return 0.0, 0.0
    
    def _random_agent_action(self, agent: Agent, dt: float) -> Tuple[float, float]:
        """Random buy/sell with Poisson arrival"""
        lambda_in = agent.parameters.get('lambda_in', 0.1)
        lambda_out = agent.parameters.get('lambda_out', 0.1)
        mean_cash_in = agent.parameters.get('mean_cash_in', 10.0)
        mean_redeem_pct = agent.parameters.get('mean_redeem_pct', 0.1)
        
        # Poisson arrivals
        n_cash_ins = self.rng.poisson(lambda_in * dt)
        n_cash_outs = self.rng.poisson(lambda_out * dt)
        
        cash_in = 0.0
        if n_cash_ins > 0 and agent.base_balance > 0:
            cash_in = min(
                self.rng.exponential(mean_cash_in) * n_cash_ins,
                agent.base_balance
            )
        
        redeem = 0.0
        if n_cash_outs > 0 and agent.token_balance > 0:
            redeem = min(
                agent.token_balance * mean_redeem_pct * n_cash_outs,
                agent.token_balance
            )
        
        return cash_in, redeem
    
    def _price_sensitive_action(
        self,
        agent: Agent,
        t: float,
        dt: float,
        P_issue: float,
        P_floor: float
    ) -> Tuple[float, float]:
        """
        Buy when price is low relative to perceived value,
        sell when price is high.
        """
        perceived_value = agent.parameters.get('perceived_value', 1.0)
        sensitivity = agent.parameters.get('sensitivity', 1.0)
        base_rate = agent.parameters.get('base_rate', 0.1)
        
        # Buy probability increases when P_issue < perceived_value
        buy_signal = sensitivity * max(0, perceived_value - P_issue) / perceived_value
        sell_signal = sensitivity * max(0, P_floor - perceived_value) / perceived_value
        
        cash_in = 0.0
        if self.rng.random() < buy_signal * dt and agent.base_balance > 0:
            amount = agent.parameters.get('trade_size', 10.0)
            cash_in = min(amount, agent.base_balance)
        
        redeem = 0.0
        if self.rng.random() < sell_signal * dt and agent.token_balance > 0:
            redeem_pct = agent.parameters.get('redeem_pct', 0.2)
            redeem = min(agent.token_balance * redeem_pct, agent.token_balance)
        
        return cash_in, redeem
    
    def _floor_trader_action(
        self,
        agent: Agent,
        t: float,
        dt: float,
        P_issue: float,
        P_floor: float
    ) -> Tuple[float, float]:
        """
        Trade based on spread between issuance price and floor.
        Buy when spread is large, sell when spread is small.
        """
        threshold = agent.parameters.get('spread_threshold', 0.2)
        trade_rate = agent.parameters.get('trade_rate', 0.05)
        
        spread = (P_issue - P_floor) / P_issue
        
        cash_in = 0.0
        if spread > threshold and agent.base_balance > 0:
            # Large spread -> buy
            prob = trade_rate * (spread - threshold)
            if self.rng.random() < prob * dt:
                amount = agent.parameters.get('trade_size', 10.0)
                cash_in = min(amount, agent.base_balance)
        
        redeem = 0.0
        if spread < threshold / 2 and agent.token_balance > 0:
            # Small spread -> sell
            prob = trade_rate * (threshold - spread)
            if self.rng.random() < prob * dt:
                redeem_pct = agent.parameters.get('redeem_pct', 0.1)
                redeem = min(agent.token_balance * redeem_pct, agent.token_balance)
        
        return cash_in, redeem
    
    def _momentum_action(
        self,
        agent: Agent,
        t: float,
        dt: float,
        P_floor: float
    ) -> Tuple[float, float]:
        """
        Follow momentum - buy when floor is rising, sell when falling.
        Requires history of floor prices.
        """
        if len(self.rate_history['t']) < 2:
            return 0.0, 0.0
        
        # Compute recent floor change
        lookback = agent.parameters.get('lookback_periods', 5)
        if len(self.rate_history['t']) < lookback:
            return 0.0, 0.0
        
        recent_floors = [self.state.marginal_floor(
            self.get_current_stage(t_hist).r_cashout, self.phi_tot
        ) for t_hist in self.rate_history['t'][-lookback:]]
        
        momentum = (recent_floors[-1] - recent_floors[0]) / recent_floors[0]
        
        sensitivity = agent.parameters.get('sensitivity', 1.0)
        
        cash_in = 0.0
        if momentum > 0 and agent.base_balance > 0:
            prob = sensitivity * momentum
            if self.rng.random() < prob * dt:
                amount = agent.parameters.get('trade_size', 10.0)
                cash_in = min(amount, agent.base_balance)
        
        redeem = 0.0
        if momentum < 0 and agent.token_balance > 0:
            prob = sensitivity * abs(momentum)
            if self.rng.random() < prob * dt:
                redeem_pct = agent.parameters.get('redeem_pct', 0.15)
                redeem = min(agent.token_balance * redeem_pct, agent.token_balance)
        
        return cash_in, redeem
    
    def _hodler_action(
        self,
        agent: Agent,
        t: float,
        dt: float,
        P_issue: float
    ) -> Tuple[float, float]:
        """
        Long-term holder: steady buying, rare selling.
        """
        buy_rate = agent.parameters.get('buy_rate', 0.05)
        sell_rate = agent.parameters.get('sell_rate', 0.001)
        
        cash_in = 0.0
        if self.rng.random() < buy_rate * dt and agent.base_balance > 0:
            amount = agent.parameters.get('regular_investment', 5.0)
            cash_in = min(amount, agent.base_balance)
        
        redeem = 0.0
        if self.rng.random() < sell_rate * dt and agent.token_balance > 0:
            # Rare emergency sell
            redeem_pct = agent.parameters.get('emergency_redeem_pct', 0.5)
            redeem = min(agent.token_balance * redeem_pct, agent.token_balance)
        
        return cash_in, redeem
    
    def _arbitrageur_action(
        self,
        agent: Agent,
        t: float,
        dt: float,
        P_issue: float,
        P_floor: float
    ) -> Tuple[float, float]:
        """
        Arbitrageur exploits mispricings between issuance and floor.
        """
        min_profit_margin = agent.parameters.get('min_profit_margin', 0.05)
        max_position_size = agent.parameters.get('max_position_size', 100.0)
        
        # Check if profitable arbitrage exists
        # Can buy at P_issue and sell at P_floor
        profit_margin = (P_floor - P_issue) / P_issue
        
        cash_in = 0.0
        redeem = 0.0
        
        if profit_margin > min_profit_margin:
            # Profitable to buy and immediately sell
            # But in practice, need to wait for execution
            # Here we model as: buy now, sell later
            if agent.base_balance > 0:
                arb_amount = min(
                    agent.parameters.get('arb_size', 50.0),
                    agent.base_balance,
                    max_position_size
                )
                cash_in = arb_amount
        
        elif agent.token_balance > 0 and profit_margin < -min_profit_margin:
            # Reverse arb not typically possible in this mechanism
            # but could model as reducing position
            redeem = min(
                agent.token_balance * 0.1,
                agent.token_balance
            )
        
        return cash_in, redeem
    
    def execute_transactions(
        self,
        t: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Execute all agent transactions for time interval [t, t+dt).
        
        Returns:
        --------
        (total_cash_in, total_redeem) : Tuple[float, float]
            Aggregate flows
        """
        stage = self.get_current_stage(t)
        P_issue = stage.issuance_price(t)
        r_cashout = stage.r_cashout
        
        total_cash_in = 0.0
        total_redeem = 0.0
        
        for agent in self.agents:
            cash_in, redeem = self.compute_agent_action(agent, t, dt)
            
            # Execute cash-in
            if cash_in > 0:
                tokens_minted = cash_in / P_issue
                agent.base_balance -= cash_in
                agent.token_balance += tokens_minted
                total_cash_in += cash_in
                
                self.transactions.append({
                    't': t,
                    'agent_id': agent.agent_id,
                    'action': 'cash_in',
                    'amount': cash_in,
                    'tokens': tokens_minted,
                    'price': P_issue
                })
            
            # Execute redemption
            if redeem > 0:
                # Compute redemption payout
                base_received = self._compute_redemption_payout(
                    redeem, r_cashout
                )
                agent.token_balance -= redeem
                agent.base_balance += base_received
                total_redeem += redeem
                
                self.transactions.append({
                    't': t,
                    'agent_id': agent.agent_id,
                    'action': 'redeem',
                    'tokens': redeem,
                    'amount': base_received,
                    'price': base_received / redeem if redeem > 0 else 0
                })
        
        # Convert to rates (amount per unit time)
        r_in = total_cash_in / dt
        r_out = total_redeem / dt
        
        return r_in, r_out
    
    def _compute_redemption_payout(
        self,
        q: float,
        r_cashout: float
    ) -> float:
        """
        Compute payout from redeeming q tokens.
        Uses the bonding curve formula.
        """
        S = self.state.S
        B = self.state.B
        
        if S <= 0 or q <= 0:
            return 0.0
        
        # C_k(q; S, B) = (q/S) * B * [(1 - r_k) + r_k * (q/S)]
        payout_pre_fee = (q / S) * B * ((1 - r_cashout) + r_cashout * (q / S))
        
        # Apply protocol fee
        payout = (1 - self.phi_tot) * payout_pre_fee
        
        return payout
    
    def simulate_step(self, t: float, dt: float):
        """
        Simulate one time step with agent actions.
        
        Updates state and records aggregate rates.
        """
        # Get aggregate flows from agents
        r_in, r_out = self.execute_transactions(t, dt)
        
        # Update system state using flows
        stage = self.get_current_stage(t)
        P_issue = stage.issuance_price(t)
        r_cashout = stage.r_cashout
        
        # Update S and B
        dS = (r_in / P_issue - r_out) * dt
        dB = (r_in - (1 - r_cashout) * (self.state.B / self.state.S) * r_out) * dt
        
        self.state.S += dS
        self.state.B += dB
        self.state.t = t + dt
        
        # Record rates
        self.rate_history['t'].append(t)
        self.rate_history['r_in'].append(r_in)
        self.rate_history['r_out'].append(r_out)
    
    def simulate(self, t_end: float, dt: float = 0.1):
        """
        Run full agent-based simulation from current time to t_end.
        
        Parameters:
        -----------
        t_end : float
            End time
        dt : float
            Time step for agent actions
        """
        t = self.state.t
        
        while t < t_end:
            actual_dt = min(dt, t_end - t)
            self.simulate_step(t, actual_dt)
            t += actual_dt
    
    def get_aggregate_rate_functions(self) -> Tuple[Callable, Callable]:
        """
        Return interpolated rate functions from agent-based simulation.
        
        Returns:
        --------
        (r_in_func, r_out_func) : Tuple[Callable, Callable]
            Functions that can be used in continuous-time ODE solver
        """
        t_array = np.array(self.rate_history['t'])
        r_in_array = np.array(self.rate_history['r_in'])
        r_out_array = np.array(self.rate_history['r_out'])
        
        def r_in_func(t: float) -> float:
            if t < t_array[0]:
                return 0.0
            if t >= t_array[-1]:
                return r_in_array[-1]
            return np.interp(t, t_array, r_in_array)
        
        def r_out_func(t: float) -> float:
            if t < t_array[0]:
                return 0.0
            if t >= t_array[-1]:
                return r_out_array[-1]
            return np.interp(t, t_array, r_out_array)
        
        return r_in_func, r_out_func


def create_agent_population(
    n_agents: int,
    agent_type_distribution: Dict[AgentType, float],
    seed: Optional[int] = None
) -> List[Agent]:
    """
    Create a heterogeneous population of agents.
    
    Parameters:
    -----------
    n_agents : int
        Total number of agents
    agent_type_distribution : Dict[AgentType, float]
        Probability distribution over agent types (must sum to 1)
    seed : Optional[int]
        Random seed
    
    Returns:
    --------
    agents : List[Agent]
        List of initialized agents
    """
    rng = np.random.RandomState(seed)
    
    # Sample agent types
    types = list(agent_type_distribution.keys())
    probs = list(agent_type_distribution.values())
    
    assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities must sum to 1"
    
    agents = []
    for i in range(n_agents):
        agent_type = rng.choice(types, p=probs)
        
        # Set default parameters based on type
        if agent_type == AgentType.RANDOM:
            params = {
                'lambda_in': rng.uniform(0.05, 0.2),
                'lambda_out': rng.uniform(0.05, 0.2),
                'mean_cash_in': rng.uniform(5, 20),
                'mean_redeem_pct': rng.uniform(0.05, 0.2)
            }
        elif agent_type == AgentType.PRICE_SENSITIVE:
            params = {
                'perceived_value': rng.uniform(0.8, 1.5),
                'sensitivity': rng.uniform(0.5, 2.0),
                'trade_size': rng.uniform(10, 30),
                'redeem_pct': rng.uniform(0.1, 0.3)
            }
        elif agent_type == AgentType.FLOOR_TRADER:
            params = {
                'spread_threshold': rng.uniform(0.1, 0.3),
                'trade_rate': rng.uniform(0.03, 0.1),
                'trade_size': rng.uniform(15, 40),
                'redeem_pct': rng.uniform(0.1, 0.25)
            }
        elif agent_type == AgentType.MOMENTUM:
            params = {
                'lookback_periods': int(rng.uniform(3, 10)),
                'sensitivity': rng.uniform(0.5, 1.5),
                'trade_size': rng.uniform(10, 25),
                'redeem_pct': rng.uniform(0.1, 0.2)
            }
        elif agent_type == AgentType.HODLER:
            params = {
                'buy_rate': rng.uniform(0.02, 0.1),
                'sell_rate': rng.uniform(0.0005, 0.005),
                'regular_investment': rng.uniform(3, 10),
                'emergency_redeem_pct': rng.uniform(0.3, 0.7)
            }
        elif agent_type == AgentType.ARBITRAGEUR:
            params = {
                'min_profit_margin': rng.uniform(0.02, 0.08),
                'arb_size': rng.uniform(30, 100),
                'max_position_size': rng.uniform(80, 200)
            }
        else:
            params = {}
        
        agent = Agent(
            agent_id=i,
            agent_type=agent_type,
            base_balance=rng.uniform(50, 200),
            parameters=params
        )
        agents.append(agent)
    
    return agents
