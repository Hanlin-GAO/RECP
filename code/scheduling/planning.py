import math
import numpy as np


def estimate_soc_after_travel(
    soc_now,
    pos_now,
    pos_goal,
    pv_env,               # 0~1
    pv_peak_kW,           # rover PV peak
    t0,
    dt_hours,
    v_per_step,
    P_load_move_kW,
    E_bat_kWh,
    eta_dis,
):
    """Estimate mean onboard solar power during travel from normalized supply times peak power. Used for travel-energy and station-selection calculations."""
    pos_now = np.asarray(pos_now, dtype=float)
    pos_goal = np.asarray(pos_goal, dtype=float)
    dist = float(np.linalg.norm(pos_goal - pos_now))
    steps = int(math.ceil(dist / max(v_per_step, 1e-6)))
    if steps <= 0:
        return float(soc_now)

    t1 = min(t0 + steps, len(pv_env))
    if t0 >= len(pv_env):
        pv_mean = 0.0
    else:
        pv_mean = float(np.mean(np.clip(pv_env[t0:t1], 0.0, 1.0))) * float(pv_peak_kW)

    deficit_mean = max(float(P_load_move_kW) - pv_mean, 0.0)
    E_need_kWh = (deficit_mean / max(float(eta_dis), 1e-9)) * steps * float(dt_hours)
    soc_next = float(soc_now) - E_need_kWh / max(float(E_bat_kWh), 1e-9)
    return float(soc_next)


def estimate_charge_steps(
    soc_arrive_est,
    soc_target,
    E_bat_kWh,
    dt_hours,
    P_ch_max_kW,
    eta_ch_rover,
    eta_wireless,
):
    """Estimate steps needed to charge from the arrival SoC to the target SoC for station reservation."""
    soc_arrive_est = float(np.clip(soc_arrive_est, 0.0, 1.0))
    soc_target = float(np.clip(soc_target, 0.0, 1.0))
    if soc_target <= soc_arrive_est + 1e-12:
        return 0

    E_need = (soc_target - soc_arrive_est) * float(E_bat_kWh)  # kWh
    # Battery energy increment is approximately eta_ch_rover * received_power * dt
    # Use the vehicle charging-power limit in the duration estimate
    effective_kW = max(float(P_ch_max_kW) * float(eta_ch_rover), 1e-9)
    steps = int(math.ceil(E_need / (effective_kW * float(dt_hours))))
    return max(steps, 1)


def sort_stations_by_distance(pos_now, stations):
    pos_now = np.asarray(pos_now, dtype=float)
    ds = []
    for st in stations:
        st_pos = np.asarray(st["pos"], dtype=float)
        ds.append((float(np.linalg.norm(st_pos - pos_now)), st))
    ds.sort(key=lambda x: x[0])
    return [x[1] for x in ds]
