import numpy as np


class StationManager:
    """
    - One rover at a time (wireless charging channel).
    - FIFO queue.
    - Station has PV + battery SoC.
    - NEW: P_tx output ramp limit to avoid sawtooth.
    """

    def __init__(self, stations, dt_hours=0.25):
        self.dt = float(dt_hours)
        self.stations = {int(st["id"]): dict(st) for st in stations}

        self.E = {}
        self.E_min = {}
        self.E_max = {}
        self.P_last = {}  # NEW: last transmit power (kW)

        for sid, st in self.stations.items():
            E_kWh = float(st.get("E_station_kWh", 150.0))
            soc0 = float(st.get("soc0_station", 0.80))
            soc_min = float(st.get("soc_min_station", 0.20))
            soc_max = float(st.get("soc_max_station", 0.95))

            self.E[sid] = soc0 * E_kWh
            self.E_min[sid] = soc_min * E_kWh
            self.E_max[sid] = soc_max * E_kWh
            self.P_last[sid] = 0.0

        # reservation bookkeeping
        self.resv = {sid: {} for sid in self.stations.keys()}  # rover_id -> (arrival_step, charge_steps)
        self.queue = {sid: [] for sid in self.stations.keys()}  # arrived waiting rovers (fifo)
        self.current = {sid: None for sid in self.stations.keys()}  # rover id being charged
        self._busy_until = {sid: -1 for sid in self.stations.keys()}  # reserved end time for "idle check"

    def station_soc(self, sid: int) -> float:
        st = self.stations[int(sid)]
        E_kWh = float(st.get("E_station_kWh", 150.0))
        return float(self.E[int(sid)] / max(E_kWh, 1e-9))

    def queue_length(self, sid: int) -> int:
        sid = int(sid)
        return int(len(self.queue[sid]) + (1 if self.current[sid] is not None else 0))

    def request_charge(self, sid: int, rover_id: int, t_request: int, arrival_step: int,
                       charge_steps: int, require_idle_at_arrival=True) -> bool:
        """
        If require_idle_at_arrival=True:
          accept only if arrival_step >= busy_until (station will be idle at arrival time)
        else:
          always accept (rover will wait at station)
        """
        sid = int(sid)
        rover_id = int(rover_id)
        arrival_step = int(arrival_step)
        charge_steps = max(int(charge_steps), 1)

        if require_idle_at_arrival:
            if arrival_step < self._busy_until[sid]:
                return False

        self.resv[sid][rover_id] = (arrival_step, charge_steps)

        # update busy horizon (rough reservation)
        end_step = arrival_step + charge_steps
        self._busy_until[sid] = max(self._busy_until[sid], end_step)

        return True

    def mark_arrived(self, sid: int, rover_id: int):
        sid = int(sid)
        rover_id = int(rover_id)
        if rover_id in self.resv[sid]:
            self.queue[sid].append(rover_id)

    def start_charging_if_possible(self, sid: int):
        sid = int(sid)
        if self.current[sid] is not None:
            return
        if len(self.queue[sid]) == 0:
            return
        rid = self.queue[sid].pop(0)
        self.current[sid] = rid

    def release(self, sid: int, rover_id: int):
        sid = int(sid)
        rover_id = int(rover_id)

        if self.current[sid] == rover_id:
            self.current[sid] = None

        if rover_id in self.resv[sid]:
            del self.resv[sid][rover_id]

    def compute_station_power_once(self, sid: int, t: int, pv_env, P_tx_request_kW: float):
        """
        Returns:
          P_tx_used_kW, unmet_kW (station side unmet for total load)
        Station power balance:
          PV + battery_discharge - (base_load + P_tx_used) -> battery_charge/curtail
        NEW:
          Apply ramp limit on P_tx_used to avoid jagged output.
        """
        sid = int(sid)
        st = self.stations[sid]

        renewable_profile = st.get("renewable_profile_kW", None)
        if renewable_profile is not None:
            renewable_profile = np.asarray(renewable_profile, dtype=float).reshape(-1)
            P_pv = float(renewable_profile[t]) if (0 <= t < len(renewable_profile)) else 0.0
        else:
            pv_env = np.asarray(pv_env, dtype=float).reshape(-1)
            pv_alpha = float(pv_env[t]) if (0 <= t < len(pv_env)) else 0.0
            pv_alpha = float(np.clip(pv_alpha, 0.0, 1.0))

            pv_peak = float(st.get("pv_peak_kW", 50.0))
            P_pv = pv_alpha * pv_peak

        P_aux = float(st.get("P_aux_kW", 0.40))

        P_tx_max = float(st.get("P_tx_max_kW", 40.0))
        P_st_ch_max = float(st.get("P_st_ch_max_kW", 25.0))
        P_st_dis_max = float(st.get("P_st_dis_max_kW", 40.0))

        # raw request clamp
        P_req = float(np.clip(P_tx_request_kW, 0.0, P_tx_max))

        # station energy margins
        E_now = self.E[sid]
        E_min = self.E_min[sid]
        E_max = self.E_max[sid]

        # available discharge (kW) without crossing E_min
        dis_kWh_cap = max(E_now - E_min, 0.0)
        P_dis_cap = min(P_st_dis_max, dis_kWh_cap / max(self.dt, 1e-9))

        # supply base load first
        need_base = max(P_aux - P_pv, 0.0)
        P_dis_for_base = min(need_base, P_dis_cap)
        P_left_dis = max(P_dis_cap - P_dis_for_base, 0.0)

        # remaining PV after base
        P_pv_after_base = max(P_pv - P_aux, 0.0)

        # available for transmit = PV_after_base + remaining discharge cap
        P_avail_for_tx = P_pv_after_base + P_left_dis

        P_tx_raw = min(P_req, P_avail_for_tx)

        # NEW: apply ramp limit
        ramp = float(st.get("P_tx_ramp_kW_per_step", 6.0))
        P_prev = float(self.P_last[sid])
        P_tx_used = float(np.clip(P_tx_raw, P_prev - ramp, P_prev + ramp))
        self.P_last[sid] = P_tx_used

        # Now compute actual battery power (charge/discharge)
        # total load seen by station after PV:
        # P_need = P_aux + P_tx_used - P_pv
        P_need = (P_aux + P_tx_used) - P_pv

        if P_need >= 0:
            # discharge to cover as much as possible
            P_dis = min(P_need, P_dis_cap)
            unmet = max(P_need - P_dis, 0.0)
            dE = -(P_dis) * self.dt
        else:
            # surplus PV, charge station battery
            P_sur = -P_need
            ch_kWh_cap = max(E_max - E_now, 0.0)
            P_ch_cap = min(P_st_ch_max, ch_kWh_cap / max(self.dt, 1e-9))
            P_ch = min(P_sur, P_ch_cap)
            unmet = 0.0
            dE = +(P_ch) * self.dt

        self.E[sid] = float(np.clip(self.E[sid] + dE, E_min, E_max))
        return P_tx_used, unmet
