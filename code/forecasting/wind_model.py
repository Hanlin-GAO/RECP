"""Wind power models. The data-driven architectures are shared with model.py; WindPINNModel adds a turbine power-curve branch."""

import torch
import torch.nn as nn

# Reuse the domain-independent model implementations
from model import DNNModel, BaselineModel, GRUModel, TCNModel, TransformerModel, _LSTMBackbone


class WindPINNModel(nn.Module):
    """Wind PINN blends a GRU with a temperature-adjusted turbine power curve.

    Below cut-in and above cut-out speeds, physical power approaches zero.
    Between cut-in and rated speed, power follows a cubic curve; above rated speed it approaches rated power.
    Sigmoid transitions and an inverse-temperature density correction are used.
    A learned gate controls the blend: (1 - gate) * GRU + gate * Physics.
    The historical trainer initializes the backbone from a trained GRU."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2,
                 rated_power_init=50.0):
        super().__init__()

        # Learnable physical parameters
        self.v_cut_in = nn.Parameter(torch.tensor(3.5))          # Cut-in wind speed in m/s
        self.v_rated = nn.Parameter(torch.tensor(11.0))          # Rated wind speed in m/s
        self.v_cut_out = nn.Parameter(torch.tensor(25.0))        # Cut-out wind speed in m/s
        self.log_k = nn.Parameter(torch.tensor(1.5))             # Transition sharpness
        self.log_P_rated = nn.Parameter(                         # Rated power
            torch.log(torch.tensor(max(rated_power_init, 1.0))))
        self.T_ref = nn.Parameter(torch.tensor(288.15))          # Reference temperature in kelvin

        # Input-dependent gate maps wind speed and temperature to [0, 1]
        # Learn the contribution of the physical branch from the supplied inputs
        self.gate_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        # Initialize the gate toward the data-driven branch (approximately 0.18)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.constant_(self.gate_net[2].bias, -1.5)

        # GRU backbone with the same architecture as GRUModel
        self.backbone = GRUModel(input_dim, hidden_dim, num_layers, dropout)

        # Training-time input dropout
        self.input_drop = nn.Dropout(p=0.15)

        # Output scaling buffers
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

    def set_output_scale(self, y_mean: float, y_std: float):
        self.y_mean.fill_(y_mean)
        self.y_std.fill_(y_std)

    def physics_forward(self, v, T_amb):
        """v: [B] wind speed (m/s), T_amb: [B] temperature (°C) → P_phys [B] (MW)"""
        k = torch.exp(self.log_k)
        P_rated = torch.exp(self.log_P_rated)

        # Partial-load region uses normalized wind speed cubed
        v_range = torch.clamp(self.v_rated - self.v_cut_in, min=0.5)
        v_norm = (v - self.v_cut_in) / v_range
        v_norm = torch.clamp(v_norm, 0.0, 1.0)
        P_partial = P_rated * v_norm.pow(3)

        # Smooth transition to rated power
        above_rated = torch.sigmoid(k * (v - self.v_rated))
        P_base = (1 - above_rated) * P_partial + above_rated * P_rated

        # Smooth cut-in and cut-out masks
        cut_in = torch.sigmoid(k * (v - self.v_cut_in))
        cut_out = torch.sigmoid(k * (self.v_cut_out - v))

        # Air-density temperature correction is proportional to inverse absolute temperature
        T_abs = T_amb + 273.15
        rho_factor = self.T_ref / T_abs

        return P_base * cut_in * cut_out * rho_factor

    def forward(self, x, v=None, T_amb=None):
        """x: normalized input [B, T, F]. v: supplied wind speed [B], m/s. T_amb: supplied temperature [B], degrees Celsius. The caller controls the information time of these weather inputs."""
        # Apply input dropout during training
        if self.training:
            x = self.input_drop(x)

        gru_out = self.backbone(x)

        if v is not None and T_amb is not None:
            P_phys = self.physics_forward(v, T_amb)
            P_phys_scaled = (P_phys - self.y_mean) / self.y_std
            # Normalize wind speed and temperature before evaluating the gate
            gate_input = torch.stack([v / 15.0, T_amb / 25.0], dim=-1)
            gate = torch.sigmoid(self.gate_net(gate_input).squeeze(-1))
            return (1 - gate) * gru_out + gate * P_phys_scaled
        else:
            return gru_out
