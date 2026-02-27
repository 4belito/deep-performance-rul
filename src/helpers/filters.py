"""Filters for smoothing degradation estimates."""

from collections import deque

import numpy as np


class AdaptiveEMA:
    def __init__(self, beta=0.05, k=10.0):
        self.beta = beta  # noise memory
        self.k = k  # smoothing strength
        self.y = None
        self.var = 0.0

    def step(self, x):
        if self.y is None:
            self.y = x
            return x

        # innovation
        e = x - self.y

        # update noise estimate (EW variance)
        self.var = self.beta * (e**2) + (1 - self.beta) * self.var
        sigma = self.var**0.5

        # adaptive smoothing
        alpha = 1.0 / (1.0 + self.k * sigma)

        # update filtered signal
        self.y = alpha * x + (1 - alpha) * self.y

        return self.y


class Kalman1D:
    def __init__(self, Q=1e-4, R=1e-2, x0=None, P0=1.0):
        self.Q = Q  # process noise variance
        self.R = R  # measurement noise variance

        self.x = x0
        self.P = P0

    def step(self, z):
        if self.x is None:
            self.x = z
            return z

        # --- Predict ---
        x_pred = self.x
        P_pred = self.P + self.Q

        # --- Update ---
        K = P_pred / (P_pred + self.R)

        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred

        return self.x


class Kalman2D:
    def __init__(self, Q_pos=1e-4, Q_vel=1e-5, R=1e-2):
        # State: [position, velocity]
        self.x = None

        # Covariance
        self.P = np.eye(2)

        # System matrices
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])

        self.H = np.array([[1.0, 0.0]])

        # Process noise
        self.Q = np.array([[Q_pos, 0.0], [0.0, Q_vel]])

        # Measurement noise
        self.R = np.array([[R]])

    def step(self, z):
        if self.x is None:
            # Initialize state
            self.x = np.array([[z], [0.0]])
            return z

        # --- Predict ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # --- Update ---
        y = np.array([[z]]) - self.H @ x_pred  # innovation
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0])


class RollingMedian:
    def __init__(self, window):
        self.window = window
        self.buffer = deque(maxlen=window)

    def step(self, x):
        self.buffer.append(x)
        return np.median(self.buffer)
