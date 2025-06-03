import numpy as np
from typing import Callable, Dict
from config import Config

class DoubleIntegrator:
    def __init__(self):
        """Initialize the double integrator system."""
        self.name = "double-integrator"
        config = Config()

        # State-space dimensions
        self.nx = config.n  # number of states
        self.nu = config.m  # number of inputs
        self.ny = config.p  # number of outputs
        
        # Create model dictionary to store system matrices and functions
        self.model = {}
        
        # Define state-space matrices
        self.model['A'] = np.array([
            [0.8147, 0.1270, 0.3324],
            [0.2058, 0.9134, 0.0975],
            [0.1270, 0.2324, 0.6785]
        ])
        
        self.model['B'] = np.array([
            [0.5469, 0.9575],
            [0.9157, 0.9649],
            [0.7577, 0.1576]
        ])
        
        self.model['C'] = np.array([
            [0.8003, 0.4218, 0.9157],
            [0.1419, 0.9157, 0.7922],
            [0.6557, 0.7922, 0.9595]
        ])
        
        self.model['D'] = np.array([
            [0.0357, 0.8491],
            [0.8491, 0.9340],
            [0.9340, 0.6787]
        ])
        
        # Define dynamics and measurement functions
        self.model['f'] = self._dynamics
        self.model['h'] = self._measurement
        
        # Constraints
        self.u_max = 1
        self.y_max = 4
        
        # Define constraint matrices
        self.constraints = {}
        
        # Input constraints
        self.constraints['Hu'] = np.vstack([
            np.eye(self.nu),
            -np.eye(self.nu)
        ])
        self.constraints['hu'] = np.ones(2 * self.nu) * self.u_max
        
        # Output constraints
        self.constraints['Hy'] = np.vstack([
            np.eye(self.ny),
            -np.eye(self.ny)
        ])
        self.constraints['hy'] = np.ones(2 * self.ny) * self.y_max
        
        # Initial conditions
        self.xs = np.array([-2.5, 1.0, 0.5])
        self.us = np.array([0.0, 0.0])
        self.ys = self._measurement(self.xs, self.us)
        
        # Equilibrium points
        self.xf = np.array([0.0, 0.0, 0.0])
        self.uf = np.array([0.0, 0.0])
        self.yf = np.array([0.0, 0.0, 0.0])
    
    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        System dynamics function f(x,u) = Ax + Bu
        
        Args:
            x: State vector
            u: Input vector
            
        Returns:
            Next state
        """
        return self.model['A'] @ x + self.model['B'] @ u
    
    def _measurement(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Measurement function h(x,u) = Cx + Du
        
        Args:
            x: State vector
            u: Input vector
            
        Returns:
            System output
        """
        return self.model['C'] @ x + self.model['D'] @ u
    
    def get_system_matrices(self) -> Dict:
        """
        Returns the system matrices A, B, C, D.
        
        Returns:
            Dictionary containing system matrices
        """
        return {
            'A': self.model['A'],
            'B': self.model['B'],
            'C': self.model['C'],
            'D': self.model['D']
        }
    
    def get_constraints(self) -> Dict:
        """
        Returns the system constraints.
        
        Returns:
            Dictionary containing constraint matrices
        """
        return self.constraints
