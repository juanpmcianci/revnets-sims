"""
Revnet Dynamical System Simulator
===================================

A continuous-time simulation environment for Revnet mechanisms with:
- Stage-based issuance pricing with discrete price steps
- Redemption bonding curve with cash-out tax
- Agent-based modeling for issuance and redemption flows
- Support for auto-issuances
- Closed-form solutions and numerical integration
- Counterfactual analysis and uncertainty quantification
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import poisson, expon
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict
import warnings


@dataclass
class Stage:
    """
    Represents a single Revnet stage with immutable parameters.
    
    Parameters:
    -----------
    t_start : float
        Start time of the stage
    P_issue_0 : float
        Initial issuance price at stage start
    gamma_cut : float
        Price cut parameter in (0, 1), determines price decay
    Delta_t : float
        Time interval between price steps
    sigma : float
        Per-mint split parameter in [0, 1)
    r_cashout : float
        Cash-out tax in [0, 1)
    auto_issuances : List[Tuple[float, float]]
        List of (time, amount) pairs for scheduled auto-issuances
    """
    t_start: float
    P_issue_0: float
    gamma_cut: float
    Delta_t: float
    sigma: float = 0.0
    r_cashout: float = 0.0
    auto_issuances: List[Tuple[float, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate stage parameters"""
        assert 0 <= self.gamma_cut < 1, "gamma_cut must be in [0, 1)"
        assert self.P_issue_0 > 0, "Initial price must be positive"
        assert self.Delta_t > 0, "Time step must be positive"
        assert 0 <= self.sigma < 1, "sigma must be in [0, 1)"
        assert 0 <= self.r_cashout < 1, "cash-out tax must be in [0, 1)"
        
        # Compute gamma (price increase factor)
        if self.gamma_cut > 0:
            self.gamma = 1 / (1 - self.gamma_cut)
        else:
            self.gamma = 1.0  # No price change
    
    def issuance_price(self, t: float) -> float:
        """
        Compute the contract issuance price at time t.
        
        P_issue(t) = P_issue_0 * gamma^floor((t - t_start) / Delta_t)
        """
        if t < self.t_start:
            raise ValueError(f"Time {t} is before stage start {self.t_start}")
        
        steps = int(np.floor((t - self.t_start) / self.Delta_t))
        return self.P_issue_0 * (self.gamma ** steps)
    
    def get_constant_price_intervals(self, t_end: float) -> List[Tuple[float, float, float]]:
        """
        Return list of (t_start, t_end, price) for intervals with constant price.
        Also accounts for auto-issuances that break intervals.
        """
        intervals = []
        t = self.t_start
        
        # Collect all event times (price steps and auto-issuances)
        event_times = set()
        
        # Add price step times
        n_steps = int(np.ceil((t_end - self.t_start) / self.Delta_t))
        for i in range(n_steps + 1):
            step_time = self.t_start + i * self.Delta_t
            if step_time <= t_end:
                event_times.add(step_time)
        
        # Add auto-issuance times
        for tau, _ in self.auto_issuances:
            if self.t_start <= tau <= t_end:
                event_times.add(tau)
        
        event_times.add(t_end)
        event_times = sorted(event_times)
        
        for i in range(len(event_times) - 1):
            t_interval_start = event_times[i]
            t_interval_end = event_times[i + 1]
            price = self.issuance_price(t_interval_start)
            intervals.append((t_interval_start, t_interval_end, price))
        
        return intervals


@dataclass
class RevnetState:
    """
    Current state of the Revnet system.
    
    Attributes:
    -----------
    B : float
        Treasury balance (base asset units)
    S : float
        Token supply (token units)
    t : float
        Current time
    """
    B: float
    S: float
    t: float = 0.0
    
    def backing_ratio(self) -> float:
        """Return B/S ratio"""
        return self.B / self.S if self.S > 0 else np.inf
    
    def marginal_floor(self, r_cashout: float, phi_tot: float = 0.0) -> float:
        """
        User-facing marginal redemption floor.
        
        P_floor = (1 - phi_tot) * (1 - r_cashout) * B / S
        """
        return (1 - phi_tot) * (1 - r_cashout) * self.backing_ratio()


class RevnetSimulator:
    """
    Main simulator for Revnet dynamical system with agent-based modeling.
    """
    
    def __init__(
        self,
        stages: List[Stage],
        initial_state: RevnetState,
        phi_tot: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the Revnet simulator.
        
        Parameters:
        -----------
        stages : List[Stage]
            Ordered list of stages (must have increasing start times)
        initial_state : RevnetState
            Initial conditions (B_0, S_0, t_0)
        phi_tot : float
            Total protocol token fee fraction in [0, 1)
        seed : Optional[int]
            Random seed for reproducibility
        """
        self.stages = sorted(stages, key=lambda s: s.t_start)
        self.state = initial_state
        self.phi_tot = phi_tot
        self.rng = np.random.RandomState(seed)
        
        # Validate stage ordering
        for i in range(len(self.stages) - 1):
            assert self.stages[i].t_start < self.stages[i + 1].t_start, \
                "Stages must have strictly increasing start times"
        
        # History tracking
        self.history = {
            't': [initial_state.t],
            'B': [initial_state.B],
            'S': [initial_state.S],
            'P_floor': [initial_state.marginal_floor(self.get_current_stage().r_cashout, phi_tot)],
            'P_issue': [self.get_current_stage().issuance_price(initial_state.t)],
        }
    
    def get_current_stage(self, t: Optional[float] = None) -> Stage:
        """Get the active stage at time t (or current time)"""
        if t is None:
            t = self.state.t
        
        for i in range(len(self.stages) - 1, -1, -1):
            if t >= self.stages[i].t_start:
                return self.stages[i]
        
        raise ValueError(f"No stage found for time {t}")
    
    def ode_system(
        self,
        t: float,
        y: np.ndarray,
        P_issue: float,
        r_cashout: float,
        r_in_func: Callable[[float], float],
        r_out_func: Callable[[float], float]
    ) -> np.ndarray:
        """
        ODE system for the Revnet dynamics on an interval with constant issuance price.
        
        dy/dt = f(t, y) where y = [S, B]
        
        dS/dt = r_in(t) / P_issue - r_out(t)
        dB/dt = r_in(t) - (1 - r_cashout) * (B/S) * r_out(t)
        """
        S, B = y
        
        if S <= 0:
            warnings.warn("Token supply reached zero during integration")
            return np.array([0.0, 0.0])
        
        r_in = r_in_func(t)
        r_out = r_out_func(t)
        
        dS_dt = r_in / P_issue - r_out
        dB_dt = r_in - (1 - r_cashout) * (B / S) * r_out
        
        return np.array([dS_dt, dB_dt])
    
    def simulate_interval(
        self,
        t_start: float,
        t_end: float,
        r_in_func: Callable[[float], float],
        r_out_func: Callable[[float], float],
        dt: float = 0.01,
        method: str = 'RK45'
    ):
        """
        Simulate the system over a time interval using numerical integration.
        
        Parameters:
        -----------
        t_start : float
            Start time
        t_end : float
            End time
        r_in_func : Callable[[float], float]
            Cash-in rate function r_in(t)
        r_out_func : Callable[[float], float]
            Cash-out rate function r_out(t)
        dt : float
            Time step for dense output
        method : str
            Integration method for solve_ivp
        """
        stage = self.get_current_stage(t_start)
        P_issue = stage.issuance_price(t_start)
        r_cashout = stage.r_cashout
        
        # Initial conditions
        y0 = np.array([self.state.S, self.state.B])
        
        # Check for auto-issuances in this interval
        auto_times = [tau for tau, _ in stage.auto_issuances if t_start < tau <= t_end]
        
        if not auto_times:
            # No auto-issuances, integrate over full interval
            sol = solve_ivp(
                lambda t, y: self.ode_system(t, y, P_issue, r_cashout, r_in_func, r_out_func),
                (t_start, t_end),
                y0,
                method=method,
                dense_output=True,
                max_step=dt
            )
            
            # Evaluate at regular time points
            t_eval = np.arange(t_start, t_end, dt)
            if t_eval[-1] < t_end:
                t_eval = np.append(t_eval, t_end)
            
            y_eval = sol.sol(t_eval)
            
            # Update state and history
            for i, t in enumerate(t_eval[1:], 1):
                S, B = y_eval[:, i]
                self.state.S = S
                self.state.B = B
                self.state.t = t
                
                self.history['t'].append(t)
                self.history['S'].append(S)
                self.history['B'].append(B)
                self.history['P_floor'].append(
                    self.state.marginal_floor(r_cashout, self.phi_tot)
                )
                self.history['P_issue'].append(stage.issuance_price(t))
        else:
            # Handle auto-issuances by breaking interval into sub-intervals
            sub_intervals = [t_start] + sorted(auto_times) + [t_end]
            
            for i in range(len(sub_intervals) - 1):
                sub_start = sub_intervals[i]
                sub_end = sub_intervals[i + 1]
                
                # Integrate to just before auto-issuance
                if sub_end in auto_times:
                    actual_end = sub_end - 1e-10
                else:
                    actual_end = sub_end
                
                # Integrate sub-interval
                y0 = np.array([self.state.S, self.state.B])
                sol = solve_ivp(
                    lambda t, y: self.ode_system(t, y, P_issue, r_cashout, r_in_func, r_out_func),
                    (sub_start, actual_end),
                    y0,
                    method=method,
                    dense_output=True,
                    max_step=dt
                )
                
                t_eval = np.arange(sub_start, actual_end, dt)
                if len(t_eval) == 0 or t_eval[-1] < actual_end:
                    t_eval = np.append(t_eval, actual_end)
                
                y_eval = sol.sol(t_eval)
                
                for j, t in enumerate(t_eval[1:], 1):
                    S, B = y_eval[:, j]
                    self.state.S = S
                    self.state.B = B
                    self.state.t = t
                    
                    self.history['t'].append(t)
                    self.history['S'].append(S)
                    self.history['B'].append(B)
                    self.history['P_floor'].append(
                        self.state.marginal_floor(r_cashout, self.phi_tot)
                    )
                    self.history['P_issue'].append(stage.issuance_price(t))
                
                # Apply auto-issuance if at event time
                if sub_end in auto_times:
                    # Find the auto-issuance amount
                    auto_amount = next(a for tau, a in stage.auto_issuances if tau == sub_end)
                    
                    # Jump: S increases, B stays same
                    self.state.S += auto_amount
                    self.state.t = sub_end
                    
                    # Record jump
                    self.history['t'].append(sub_end)
                    self.history['S'].append(self.state.S)
                    self.history['B'].append(self.state.B)
                    self.history['P_floor'].append(
                        self.state.marginal_floor(r_cashout, self.phi_tot)
                    )
                    self.history['P_issue'].append(stage.issuance_price(sub_end))
    
    def simulate(
        self,
        t_end: float,
        r_in_func: Callable[[float], float],
        r_out_func: Callable[[float], float],
        dt: float = 0.01,
        method: str = 'RK45'
    ):
        """
        Simulate the full system from current time to t_end.
        
        Automatically handles stage transitions, price steps, and auto-issuances.
        """
        t_current = self.state.t
        
        while t_current < t_end:
            stage = self.get_current_stage(t_current)
            
            # Find next stage start or t_end
            next_stage_start = None
            for s in self.stages:
                if s.t_start > t_current:
                    next_stage_start = s.t_start
                    break
            
            if next_stage_start is None or next_stage_start > t_end:
                interval_end = t_end
            else:
                interval_end = next_stage_start
            
            # Get constant-price sub-intervals within this stage
            intervals = stage.get_constant_price_intervals(interval_end)
            intervals = [(t_s, t_e, p) for t_s, t_e, p in intervals if t_s >= t_current]
            
            for t_int_start, t_int_end, price in intervals:
                if t_int_start >= interval_end:
                    break
                
                actual_end = min(t_int_end, interval_end)
                self.simulate_interval(
                    t_int_start, actual_end, r_in_func, r_out_func, dt, method
                )
            
            t_current = interval_end
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Return simulation history as numpy arrays"""
        return {k: np.array(v) for k, v in self.history.items()}
    
    def reset(self, initial_state: Optional[RevnetState] = None):
        """Reset simulator to initial or specified state"""
        if initial_state is None:
            initial_state = RevnetState(
                B=self.history['B'][0],
                S=self.history['S'][0],
                t=self.history['t'][0]
            )
        
        self.state = initial_state
        stage = self.get_current_stage(initial_state.t)
        self.history = {
            't': [initial_state.t],
            'B': [initial_state.B],
            'S': [initial_state.S],
            'P_floor': [initial_state.marginal_floor(stage.r_cashout, self.phi_tot)],
            'P_issue': [stage.issuance_price(initial_state.t)],
        }


def constant_rate(rate: float) -> Callable[[float], float]:
    """Helper function to create constant rate function"""
    return lambda t: rate


def piecewise_rate(
    time_points: List[float],
    rates: List[float]
) -> Callable[[float], float]:
    """
    Create piecewise constant rate function.
    
    Parameters:
    -----------
    time_points : List[float]
        Breakpoints for rate changes (must be sorted)
    rates : List[float]
        Rate values (len(rates) = len(time_points) + 1)
    """
    def rate_func(t: float) -> float:
        for i, tp in enumerate(time_points):
            if t < tp:
                return rates[i]
        return rates[-1]
    
    return rate_func


def sinusoidal_rate(
    base_rate: float,
    amplitude: float,
    period: float,
    phase: float = 0.0
) -> Callable[[float], float]:
    """Create sinusoidal rate function for modeling periodic behavior"""
    omega = 2 * np.pi / period
    return lambda t: max(0, base_rate + amplitude * np.sin(omega * t + phase))
