import torch
import torch.nn as nn


class DNNModel(nn.Module):
    """DNN: flatten the input window and apply a feed-forward network."""

    def __init__(self, input_dim, hidden_dim=64, window_size=24, dropout=0.2):
        super().__init__()
        flat_dim = window_size * input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, **kwargs):
        # x: [B, T, F] -> [B, T*F]
        return self.net(x.reshape(x.size(0), -1)).squeeze(-1)


class _LSTMBackbone(nn.Module):
    """Shared LSTM, attention, and MLP backbone for the baseline and PV PINN."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Use the final hidden state as the attention query over all time steps
        self.attn_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_scale = hidden_dim ** 0.5
        # Skip connection: project last-timestep raw features into the head
        self.skip_proj = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out, _ = self.lstm(x)           # [B, T, H]
        h_last = out[:, -1:, :]         # [B, 1, H]
        # attention
        q = self.attn_query(h_last)     # [B, 1, H]
        k = self.attn_key(out)          # [B, T, H]
        scores = torch.bmm(q, k.transpose(1, 2)) / self.attn_scale   # [B, 1, T]
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, out).squeeze(1)   # [B, H]
        h_last = self.layer_norm(h_last.squeeze(1))     # [B, H]
        # skip connection from last-timestep raw features
        x_last = self.skip_proj(x[:, -1, :])            # [B, H]
        combined = torch.cat([h_last, context, x_last], dim=-1) # [B, 3H]
        return self.head(combined).squeeze(-1)


class BaselineModel(nn.Module):
    """LSTM baseline with a normalized scalar prediction."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.backbone = _LSTMBackbone(input_dim, hidden_dim, num_layers, dropout)

    def forward(self, x, **kwargs):
        return self.backbone(x)


class GRUModel(nn.Module):
    """GRU with attention and a regression head."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attn_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_scale = hidden_dim ** 0.5
        self.skip_proj = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, **kwargs):
        out, _ = self.gru(x)             # [B, T, H]
        h_last = out[:, -1:, :]          # [B, 1, H]
        q = self.attn_query(h_last)
        k = self.attn_key(out)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.attn_scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, out).squeeze(1)
        h_last = self.layer_norm(h_last.squeeze(1))
        x_last = self.skip_proj(x[:, -1, :])
        combined = torch.cat([h_last, context, x_last], dim=-1)
        return self.head(combined).squeeze(-1)


class _TCNBlock(nn.Module):
    """TCN residual block with causal dilated convolutions, batch normalization, GELU, and dropout."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.trim1 = padding
        self.trim2 = padding

    def forward(self, x):
        # x: [B, C, T]
        res = self.downsample(x)
        out = self.conv1(x)
        if self.trim1 > 0:
            out = out[:, :, :-self.trim1]
        out = self.drop(nn.functional.gelu(self.bn1(out)))
        out = self.conv2(out)
        if self.trim2 > 0:
            out = out[:, :, :-self.trim2]
        out = self.drop(nn.functional.gelu(self.bn2(out)))
        return nn.functional.gelu(out + res)


class TCNModel(nn.Module):
    """Temporal convolutional network with dilated causal convolutions."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2,
                 kernel_size=3):
        super().__init__()
        layers = []
        ch_in = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            ch_out = hidden_dim
            layers.append(_TCNBlock(ch_in, ch_out, kernel_size, dilation, dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.skip_proj = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, **kwargs):
        # x: [B, T, F] → [B, F, T] for Conv1d
        out = self.tcn(x.transpose(1, 2))  # [B, H, T]
        h = out[:, :, -1]  # Use the final time step
        x_last = self.skip_proj(x[:, -1, :])  # [B, H]
        combined = torch.cat([h, x_last], dim=-1)  # [B, 2H]
        return self.head(combined).squeeze(-1)


class PINNModel(nn.Module):
    """PV PINN combines a learnable physical branch with an LSTM correction.

    P = alpha * G * clamp(1 - beta * (T_cell - T_ref), min=0)
    T_cell = T_amb + noct_coeff * G

    The gated physical prediction and the temporal correction share the normalized output scale."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        # Learnable physical parameters; log parameters enforce positivity
        self.log_alpha = nn.Parameter(torch.tensor(3.5))        # exp(3.5) ≈ 33
        self.beta = nn.Parameter(torch.tensor(0.004))            # Temperature coefficient per degree Celsius
        self.T_ref = nn.Parameter(torch.tensor(25.0))            # Reference temperature
        self.log_noct = nn.Parameter(torch.tensor(-3.5))         # NOCT heating coefficient

        # Learnable physics contribution gate; sigmoid(0) = 0.5
        self.gate_param = nn.Parameter(torch.tensor(0.0))

        # LSTM branch with the same backbone as the baseline
        self.backbone = _LSTMBackbone(input_dim, hidden_dim, num_layers, dropout)

        # Output scaling buffers set by set_output_scale
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

    def set_output_scale(self, y_mean: float, y_std: float):
        self.y_mean.fill_(y_mean)
        self.y_std.fill_(y_std)

    def physics_forward(self, G, T_amb):
        """Map irradiance and temperature vectors [B] to physical power [B] in watts."""
        alpha = torch.exp(self.log_alpha)
        noct = torch.exp(self.log_noct)
        T_cell = T_amb + noct * G
        temp_factor = torch.clamp(1.0 - self.beta * (T_cell - self.T_ref), min=0.0)
        return alpha * G * temp_factor

    def forward(self, x, G=None, T_amb=None):
        """x: normalized input [B, T, F]. G: supplied irradiance [B], W/m^2. T_amb: supplied air temperature [B], degrees Celsius. Return normalized predictions [B]. The caller determines whether weather is observed, forecast, or historical target-time input."""
        correction = self.backbone(x)  # LSTM correction in normalized target units

        if G is not None and T_amb is not None:
            P_phys = self.physics_forward(G, T_amb)
            P_phys_scaled = (P_phys - self.y_mean) / self.y_std
            gate = torch.sigmoid(self.gate_param)
            return gate * P_phys_scaled + correction
        else:
            return correction


# =============================================================================
# Transformer Encoder Model
# =============================================================================
class TransformerModel(nn.Module):
    """
    Transformer Encoder for time-series regression.

    Architecture:
        input_proj → learnable positional embedding → TransformerEncoder (Pre-LN)
        → last-timestep token + skip projection → head MLP → scalar output

    Input:  x [B, T, F]
    Output: scalar [B]
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2,
                 nhead=4, dim_feedforward=None):
        super().__init__()
        # Ensure hidden_dim is divisible by nhead
        if hidden_dim % nhead != 0:
            nhead = 1
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable positional embedding (supports up to 512 time steps)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer encoder with Pre-LN (more stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Skip connection from last raw time-step features
        self.skip_proj = nn.Linear(input_dim, hidden_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.skip_proj.weight)
        nn.init.constant_(self.skip_proj.bias, 0.0)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, **kwargs):
        B, T, _ = x.shape
        h = self.input_proj(x)            # [B, T, H]
        h = h + self.pos_emb[:, :T, :]   # add positional embedding
        h = self.transformer(h)           # [B, T, H]
        h = self.layer_norm(h)
        h_last = h[:, -1, :]             # [B, H] — last time-step token
        x_last = self.skip_proj(x[:, -1, :])  # [B, H] — raw feature skip
        combined = torch.cat([h_last, x_last], dim=-1)  # [B, 2H]
        return self.head(combined).squeeze(-1)
