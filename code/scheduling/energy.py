import numpy as np


def clamp(x, lo, hi):
    return float(np.clip(x, lo, hi))


def battery_update(
    E_kWh,
    P_source_kW,
    P_load_kW,
    dt_hours,
    E_min_kWh,
    E_max_kWh,
    eta_ch=0.95,
    eta_dis=0.95,
    P_ch_max_kW=12.0,
    P_dis_max_kW=15.0,
):
    """Update one-step power balance. Sources are onboard generation and station power. The battery covers deficits or absorbs surplus subject to power and energy limits. Return next energy, battery power (positive discharge), unmet load, charging power, and discharge power."""
    deficit = max(P_load_kW - P_source_kW, 0.0)
    surplus = max(P_source_kW - P_load_kW, 0.0)

    P_dis = min(deficit, P_dis_max_kW)
    P_ch = min(surplus, P_ch_max_kW)

    # Energy bounds
    if E_kWh <= E_min_kWh + 1e-12:
        P_dis = 0.0
    if E_kWh >= E_max_kWh - 1e-12:
        P_ch = 0.0

    dE = (eta_ch * P_ch - (P_dis / max(eta_dis, 1e-9))) * dt_hours

    # Limit discharge to preserve the lower energy bound
    if dE < 0 and (E_kWh + dE < E_min_kWh):
        allowable_dis_kWh = max(E_kWh - E_min_kWh, 0.0)
        allowable_dis_kW = allowable_dis_kWh / max(dt_hours, 1e-9) * eta_dis
        P_dis = min(P_dis, allowable_dis_kW)
        dE = (eta_ch * P_ch - (P_dis / max(eta_dis, 1e-9))) * dt_hours

    E_next = clamp(E_kWh + dE, E_min_kWh, E_max_kWh)
    P_bat = P_dis - P_ch
    unmet = max(P_load_kW - (P_source_kW + P_dis), 0.0)
    return E_next, P_bat, unmet, P_ch, P_dis


def compute_explore_load(
    mode: str,
    P_idle_kW: float,
    P_task_kW: float,
    dist_moved: float,
    P_drive_explore_kW: float,
    k_move_return: float,
    k_move_go_charge: float,
):
    """Exploration load model. Working includes task, travel, and idle power; station travel and return use distance-dependent motion power; charging and waiting use idle power."""
    mode = str(mode)

    if mode == "EXPLORE":
        return float(P_idle_kW + P_task_kW + P_drive_explore_kW)

    if mode == "GO_CHARGE":
        return float(P_idle_kW + k_move_go_charge * max(dist_moved, 0.0))

    if mode == "RETURN_RESUME":
        return float(P_idle_kW + k_move_return * max(dist_moved, 0.0))

    if mode in ["CHARGE", "WAIT", "DONE"]:
        return float(P_idle_kW)

    return float(P_idle_kW)
