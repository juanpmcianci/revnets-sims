"""
Analysis and Uncertainty Quantification for Revnet
===================================================

Tools for:
- Counterfactual analysis
- Sensitivity analysis
- Uncertainty quantification
- Parameter estimation
- Steady-state analysis
"""

import numpy as np
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import qmc
from scipy.optimize import minimize, root_scalar
import warnings

from revnet_simulator import (
    RevnetSimulator, RevnetState, Stage, constant_rate, piecewise_rate
)


@dataclass
class SteadyState:
    """
    Steady state solution for constant rates and issuance price.
    """
    r_in: float
    r_out: float
    P_issue: float
    r_cashout: float
    phi_tot: float
    
    def __post_init__(self):
        """Compute steady state values"""
        # r_out^* = r_in / P_issue
        self.r_out_star = self.r_in / self.P_issue
        
        # B*/S* = P_issue / (1 - r_cashout)
        self.backing_ratio = self.P_issue / (1 - self.r_cashout)
        
        # P_floor^* = (1 - phi_tot) * P_issue
        self.floor_steady = (1 - self.phi_tot) * self.P_issue
    
    def is_at_steady_state(
        self,
        state: RevnetState,
        r_in: float,
        r_out: float,
        tol: float = 0.01
    ) -> bool:
        """Check if system is close to steady state"""
        actual_ratio = state.backing_ratio()
        ratio_close = abs(actual_ratio - self.backing_ratio) / self.backing_ratio < tol
        
        rate_close = abs(r_out - self.r_out_star) / (self.r_out_star + 1e-10) < tol
        
        return ratio_close and rate_close


class AnalyticalSolver:
    """
    Compute analytical solutions for the Revnet ODE system.
    """
    
    @staticmethod
    def solve_constant_rates(
        B_0: float,
        S_0: float,
        r_in: float,
        r_out: float,
        P_issue: float,
        r_cashout: float,
        t: float
    ) -> Tuple[float, float]:
        """
        Analytical solution for constant rates over time interval [0, t].
        
        Uses formulas from equations (15)-(17) in the document.
        
        Returns:
        --------
        (S(t), B(t)) : Tuple[float, float]
        """
        # Net supply drift
        beta = r_in / P_issue - r_out
        
        # S evolves linearly
        S_t = S_0 + beta * t
        
        if S_t <= 0:
            warnings.warn(f"Supply became non-positive: S({t}) = {S_t}")
            return max(S_t, 1e-10), 0.0
        
        # For B, use formula (16)
        if abs(beta) < 1e-10:
            # beta ≈ 0: exponential convergence
            alpha = (1 - r_cashout) * r_out / S_0
            B_steady = r_in / alpha if alpha > 0 else B_0
            B_t = B_steady + (B_0 - B_steady) * np.exp(-alpha * t)
        else:
            # General case
            gamma = (1 - r_cashout) * r_out / beta
            ratio = S_t / S_0
            
            # B(t) = (S(t)/S_0)^{-gamma} * B_0 + [r_in * S_0 / (beta*(1+gamma))] * [S(t)/S_0 - (S(t)/S_0)^{-gamma}]
            term1 = (ratio ** (-gamma)) * B_0
            term2 = (r_in * S_0 / (beta * (1 + gamma))) * (ratio - ratio ** (-gamma))
            B_t = term1 + term2
        
        return S_t, B_t
    
    @staticmethod
    def backing_ratio_evolution(
        y_0: float,
        r_in: float,
        r_out: float,
        P_issue: float,
        r_cashout: float,
        t: float,
        S_0: float = 1.0
    ) -> float:
        """
        Analytical solution for backing ratio y(t) = B(t)/S(t).
        
        Uses formula (17) from document.
        """
        beta = r_in / P_issue - r_out
        
        if abs(beta) < 1e-10:
            # Steady case
            alpha = (1 - r_cashout) * r_out / S_0
            y_steady = r_in / (alpha * S_0) if alpha > 0 else y_0
            y_t = y_steady + (y_0 - y_steady) * np.exp(-alpha * t)
        else:
            # General case
            gamma = (1 - r_cashout) * r_out / beta
            
            # y_infty = r_in / (r_in/P_issue - r_cashout * r_out)
            denominator = r_in / P_issue - r_cashout * r_out
            if abs(denominator) < 1e-10:
                y_infty = y_0
            else:
                y_infty = r_in / denominator
            
            # Ratio S(t)/S_0
            ratio = 1 + beta * t / S_0
            
            if ratio <= 0:
                return y_0
            
            # y(t) = y_infty + (ratio)^{-(1+gamma)} * (y_0 - y_infty)
            y_t = y_infty + (ratio ** (-(1 + gamma))) * (y_0 - y_infty)
        
        return y_t
    
    @staticmethod
    def floor_derivative(
        B: float,
        S: float,
        r_in: float,
        r_out: float,
        P_issue: float,
        r_cashout: float
    ) -> float:
        """
        Compute instantaneous rate of change of marginal floor.
        
        Uses formula (14) from document:
        dP_floor/dt = [(1-r_k)/S] * [r_in * (S - B/P_issue) + r_k * B * r_out]
        """
        if S <= 0:
            return 0.0
        
        factor = (1 - r_cashout) / S
        term1 = r_in * (S - B / P_issue)
        term2 = r_cashout * B * r_out
        
        return factor * (term1 + term2)
    
    @staticmethod
    def neutral_line_r_out(
        r_in: float,
        B: float,
        S: float,
        P_issue: float,
        r_cashout: float
    ) -> float:
        """
        Compute r_out on the neutral line (where floor is stationary).
        
        From equation (19):
        r_in * [S - B/P_issue] + r_k * B * r_out = 0
        """
        if abs(r_cashout * B) < 1e-10:
            return np.inf
        
        r_out_neutral = -r_in * (S - B / P_issue) / (r_cashout * B)
        return max(0, r_out_neutral)


class CounterfactualAnalyzer:
    """
    Perform counterfactual analysis: what if parameters/rates were different?
    """
    
    def __init__(
        self,
        stages: List[Stage],
        initial_state: RevnetState,
        phi_tot: float = 0.0,
        seed: Optional[int] = None
    ):
        self.stages = stages
        self.initial_state = initial_state
        self.phi_tot = phi_tot
        self.seed = seed
    
    def compare_rate_scenarios(
        self,
        t_end: float,
        rate_scenarios: Dict[str, Tuple[Callable, Callable]],
        dt: float = 0.01
    ) -> Dict[str, Dict]:
        """
        Compare multiple rate scenarios.
        
        Parameters:
        -----------
        t_end : float
            Simulation end time
        rate_scenarios : Dict[str, Tuple[Callable, Callable]]
            Dictionary mapping scenario names to (r_in_func, r_out_func) tuples
        dt : float
            Time step
        
        Returns:
        --------
        results : Dict[str, Dict]
            Dictionary mapping scenario names to simulation results
        """
        results = {}
        
        for scenario_name, (r_in_func, r_out_func) in rate_scenarios.items():
            sim = RevnetSimulator(
                self.stages,
                RevnetState(self.initial_state.B, self.initial_state.S, self.initial_state.t),
                self.phi_tot,
                self.seed
            )
            
            sim.simulate(t_end, r_in_func, r_out_func, dt)
            
            history = sim.get_history()
            results[scenario_name] = {
                'history': history,
                'final_B': history['B'][-1],
                'final_S': history['S'][-1],
                'final_floor': history['P_floor'][-1],
                'floor_growth': (history['P_floor'][-1] - history['P_floor'][0]) / history['P_floor'][0],
                'treasury_growth': (history['B'][-1] - history['B'][0]) / history['B'][0],
            }
        
        return results
    
    def parameter_sweep(
        self,
        t_end: float,
        param_name: str,
        param_values: List[float],
        r_in_func: Callable,
        r_out_func: Callable,
        dt: float = 0.01
    ) -> Dict[float, Dict]:
        """
        Sweep a stage parameter and observe outcomes.
        
        Parameters:
        -----------
        param_name : str
            Name of parameter to sweep ('r_cashout', 'gamma_cut', 'P_issue_0')
        param_values : List[float]
            Values to test
        r_in_func, r_out_func : Callable
            Rate functions
        
        Returns:
        --------
        results : Dict[float, Dict]
            Dictionary mapping parameter values to outcomes
        """
        results = {}
        
        for value in param_values:
            # Create modified stages
            stages_modified = []
            for stage in self.stages:
                stage_dict = {
                    't_start': stage.t_start,
                    'P_issue_0': stage.P_issue_0,
                    'gamma_cut': stage.gamma_cut,
                    'Delta_t': stage.Delta_t,
                    'sigma': stage.sigma,
                    'r_cashout': stage.r_cashout,
                    'auto_issuances': stage.auto_issuances
                }
                stage_dict[param_name] = value
                stages_modified.append(Stage(**stage_dict))
            
            sim = RevnetSimulator(
                stages_modified,
                RevnetState(self.initial_state.B, self.initial_state.S, self.initial_state.t),
                self.phi_tot,
                self.seed
            )
            
            sim.simulate(t_end, r_in_func, r_out_func, dt)
            
            history = sim.get_history()
            results[value] = {
                'history': history,
                'final_floor': history['P_floor'][-1],
                'final_backing_ratio': history['B'][-1] / history['S'][-1],
                'avg_floor': np.mean(history['P_floor']),
            }
        
        return results


class UncertaintyQuantifier:
    """
    Perform uncertainty quantification through Monte Carlo sampling.
    """
    
    def __init__(
        self,
        stages: List[Stage],
        initial_state: RevnetState,
        phi_tot: float = 0.0,
        seed: Optional[int] = None
    ):
        self.stages = stages
        self.initial_state = initial_state
        self.phi_tot = phi_tot
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def monte_carlo_rates(
        self,
        t_end: float,
        r_in_sampler: Callable[[np.random.RandomState], Callable],
        r_out_sampler: Callable[[np.random.RandomState], Callable],
        n_samples: int = 100,
        dt: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulations with uncertain rate functions.
        
        Parameters:
        -----------
        r_in_sampler : Callable
            Function that takes RNG and returns a rate function
        r_out_sampler : Callable
            Function that takes RNG and returns a rate function
        n_samples : int
            Number of Monte Carlo samples
        
        Returns:
        --------
        results : Dict[str, np.ndarray]
            Arrays of shape (n_samples, n_timesteps) for each metric
        """
        all_histories = []
        
        for i in range(n_samples):
            seed_i = self.seed + i if self.seed is not None else None
            rng_i = np.random.RandomState(seed_i)
            
            r_in_func = r_in_sampler(rng_i)
            r_out_func = r_out_sampler(rng_i)
            
            sim = RevnetSimulator(
                self.stages,
                RevnetState(self.initial_state.B, self.initial_state.S, self.initial_state.t),
                self.phi_tot,
                seed_i
            )
            
            sim.simulate(t_end, r_in_func, r_out_func, dt)
            all_histories.append(sim.get_history())
        
        # Combine results
        # Assume all histories have same time points (or interpolate)
        n_times = len(all_histories[0]['t'])
        
        results = {
            'B': np.zeros((n_samples, n_times)),
            'S': np.zeros((n_samples, n_times)),
            'P_floor': np.zeros((n_samples, n_times)),
            't': all_histories[0]['t']
        }
        
        for i, hist in enumerate(all_histories):
            results['B'][i, :] = hist['B']
            results['S'][i, :] = hist['S']
            results['P_floor'][i, :] = hist['P_floor']
        
        # Compute statistics
        results['B_mean'] = np.mean(results['B'], axis=0)
        results['B_std'] = np.std(results['B'], axis=0)
        results['B_q05'] = np.percentile(results['B'], 5, axis=0)
        results['B_q95'] = np.percentile(results['B'], 95, axis=0)
        
        results['P_floor_mean'] = np.mean(results['P_floor'], axis=0)
        results['P_floor_std'] = np.std(results['P_floor'], axis=0)
        results['P_floor_q05'] = np.percentile(results['P_floor'], 5, axis=0)
        results['P_floor_q95'] = np.percentile(results['P_floor'], 95, axis=0)
        
        return results
    
    def latin_hypercube_sampling(
        self,
        t_end: float,
        param_distributions: Dict[str, Tuple[float, float]],
        base_r_in: float,
        base_r_out: float,
        n_samples: int = 100,
        dt: float = 0.01
    ) -> Dict:
        """
        Perform Latin Hypercube Sampling over parameter space.
        
        Parameters:
        -----------
        param_distributions : Dict[str, Tuple[float, float]]
            Dictionary mapping parameter names to (min, max) bounds
        base_r_in, base_r_out : float
            Base rate values
        n_samples : int
            Number of LHS samples
        
        Returns:
        --------
        results : Dict
            Contains 'samples' (parameter values) and 'outcomes' (metrics)
        """
        # Set up LHS sampler
        param_names = list(param_distributions.keys())
        n_params = len(param_names)
        
        sampler = qmc.LatinHypercube(d=n_params, seed=self.seed)
        unit_samples = sampler.random(n=n_samples)
        
        # Scale to parameter ranges
        param_samples = np.zeros_like(unit_samples)
        for i, param_name in enumerate(param_names):
            lb, ub = param_distributions[param_name]
            param_samples[:, i] = lb + (ub - lb) * unit_samples[:, i]
        
        # Run simulations
        outcomes = []
        
        for i in range(n_samples):
            # Create stages with sampled parameters
            stages_sampled = []
            for stage in self.stages:
                stage_dict = {
                    't_start': stage.t_start,
                    'P_issue_0': stage.P_issue_0,
                    'gamma_cut': stage.gamma_cut,
                    'Delta_t': stage.Delta_t,
                    'sigma': stage.sigma,
                    'r_cashout': stage.r_cashout,
                    'auto_issuances': stage.auto_issuances
                }
                
                for j, param_name in enumerate(param_names):
                    if param_name in stage_dict:
                        stage_dict[param_name] = param_samples[i, j]
                
                stages_sampled.append(Stage(**stage_dict))
            
            # Modify rates if they are parameters
            r_in = base_r_in
            r_out = base_r_out
            if 'r_in' in param_names:
                idx = param_names.index('r_in')
                r_in = param_samples[i, idx]
            if 'r_out' in param_names:
                idx = param_names.index('r_out')
                r_out = param_samples[i, idx]
            
            sim = RevnetSimulator(
                stages_sampled,
                RevnetState(self.initial_state.B, self.initial_state.S, self.initial_state.t),
                self.phi_tot,
                self.seed
            )
            
            sim.simulate(t_end, constant_rate(r_in), constant_rate(r_out), dt)
            history = sim.get_history()
            
            outcomes.append({
                'final_floor': history['P_floor'][-1],
                'final_backing_ratio': history['B'][-1] / history['S'][-1],
                'floor_volatility': np.std(np.diff(history['P_floor'])),
                'treasury_growth': (history['B'][-1] - history['B'][0]) / history['B'][0],
            })
        
        return {
            'param_names': param_names,
            'param_samples': param_samples,
            'outcomes': outcomes
        }


class SensitivityAnalyzer:
    """
    Perform sensitivity analysis using variance-based methods.
    """
    
    @staticmethod
    def compute_sobol_indices(
        lhs_results: Dict,
        outcome_key: str = 'final_floor'
    ) -> Dict[str, float]:
        """
        Compute first-order Sobol' indices (approximation).
        
        Measures how much each parameter contributes to output variance.
        """
        param_samples = lhs_results['param_samples']
        outcomes = np.array([o[outcome_key] for o in lhs_results['outcomes']])
        param_names = lhs_results['param_names']
        
        n_params = len(param_names)
        indices = {}
        
        # Total variance
        total_var = np.var(outcomes)
        
        if total_var < 1e-10:
            return {name: 0.0 for name in param_names}
        
        # Compute correlation-based approximation
        for i, param_name in enumerate(param_names):
            corr = np.corrcoef(param_samples[:, i], outcomes)[0, 1]
            # First-order approximation: S_i ≈ corr^2
            indices[param_name] = corr ** 2
        
        return indices
    
    @staticmethod
    def local_sensitivity(
        stages: List[Stage],
        initial_state: RevnetState,
        r_in: float,
        r_out: float,
        param_name: str,
        base_value: float,
        t_end: float,
        phi_tot: float = 0.0,
        epsilon: float = 0.01,
        dt: float = 0.01
    ) -> float:
        """
        Compute local sensitivity: ∂(final_floor)/∂(param) using finite differences.
        """
        # Baseline simulation
        stages_base = []
        for stage in stages:
            stage_dict = {
                't_start': stage.t_start,
                'P_issue_0': stage.P_issue_0,
                'gamma_cut': stage.gamma_cut,
                'Delta_t': stage.Delta_t,
                'sigma': stage.sigma,
                'r_cashout': stage.r_cashout,
                'auto_issuances': stage.auto_issuances
            }
            stages_base.append(Stage(**stage_dict))
        
        sim_base = RevnetSimulator(stages_base, initial_state, phi_tot)
        sim_base.simulate(t_end, constant_rate(r_in), constant_rate(r_out), dt)
        outcome_base = sim_base.get_history()['P_floor'][-1]
        
        # Perturbed simulation
        stages_pert = []
        for stage in stages:
            stage_dict = {
                't_start': stage.t_start,
                'P_issue_0': stage.P_issue_0,
                'gamma_cut': stage.gamma_cut,
                'Delta_t': stage.Delta_t,
                'sigma': stage.sigma,
                'r_cashout': stage.r_cashout,
                'auto_issuances': stage.auto_issuances
            }
            stage_dict[param_name] = base_value * (1 + epsilon)
            stages_pert.append(Stage(**stage_dict))
        
        sim_pert = RevnetSimulator(stages_pert, initial_state, phi_tot)
        sim_pert.simulate(t_end, constant_rate(r_in), constant_rate(r_out), dt)
        outcome_pert = sim_pert.get_history()['P_floor'][-1]
        
        # Finite difference
        sensitivity = (outcome_pert - outcome_base) / (base_value * epsilon)
        
        return sensitivity
