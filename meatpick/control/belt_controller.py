"""Adaptive belt speed: slow the belt when the arm falls behind, so bursts do not become misses.

Slack is how long the next piece will stay reachable after the arm becomes free:

    slack = (time until the next piece leaves the reach zone) - (time until the arm is free)

Slowing the belt stretches the first term. A PI controller holds slack near a setpoint (about the
arm's move time plus a margin). When a miss is imminent the belt is stopped, but only until the
arm is free again, so the stop is brief and cannot chatter. Belt acceleration is limited (the
world enforces 0.5 m/s^2) so pieces do not slide; the controller also exposes the speed profile it
expects so the intercept planner can aim at where pieces will really be.
"""

from __future__ import annotations

from collections.abc import Iterable

from meatpick.types import Track


class BeltController:
    def __init__(self, cfg: dict):
        c, b = cfg["control"], cfg["belt"]
        self.enabled = c["adaptive_belt"]
        self.setpoint = c["slack_setpoint_s"]
        self.clip = c["slack_clip_s"]
        self.kp, self.ki, self.i_max = c["pi_kp"], c["pi_ki"], c["pi_integral_max"]
        self.stop_slack, self.stop_horizon = c["stop_slack_s"], c["stop_horizon_s"]
        self.v_nom = b["speed_nominal_mps"]
        self.v_hi = min(c["belt_speed_max_mps"], b["speed_max_mps"])
        self.v_lo = b["speed_min_mps"]
        self.accel = b["accel_max_mps2"]
        self.mm_per_count = b["mm_per_encoder_count"]
        self.zone_hi = cfg["robot"]["reach_zone_x_mm"][1]
        self.integral = 0.0
        self.stopped = False  # latched: see update()
        self._stop_phase = 0  # 1: waiting for the arm to finish; 2: waiting for it to grip
        self._idle_s = 0.0  # how long the arm has sat free while we hold the stop
        self.command_mps = self.v_nom
        self.speed_mps = self.v_nom  # last measured belt speed
        self.stop_count = 0
        self.last_slack = float("inf")

    # --- the control law ----------------------------------------------------------------
    def next_piece_slack(
        self, now_s: float, encoder_now: int, tracks: Iterable[Track], skip: set[int],
        arm_free_at_s: float, speed_mps: float,
    ) -> tuple[float, float]:  # fmt: skip
        """(slack, time until the deadline) for the pending piece with the earliest deadline."""
        v = max(speed_mps * 1000.0, 1.0)  # mm/s, floored so a stopped belt is not "infinite"
        best = None
        for t in tracks:
            if t.status != "confirmed" or t.id in skip:
                continue
            x = t.x_mm + (encoder_now - t.ref_encoder) * self.mm_per_count
            if x > self.zone_hi:
                continue  # already gone
            d_in = (self.zone_hi - x) / v
            if best is None or d_in < best:
                best = d_in
        if best is None:
            return float("inf"), float("inf")
        return best - max(arm_free_at_s - now_s, 0.0), best

    def update(
        self,
        now_s: float,
        dt_s: float,
        encoder_now: int,
        tracks: Iterable[Track],
        skip: set[int],
        arm,
        speed_mps: float,
    ) -> float:
        """Belt speed command in m/s. ``skip`` holds track ids that are issued or missed."""
        self.speed_mps = speed_mps
        if not self.enabled:
            self.command_mps = self.v_nom
            return self.command_mps
        slack, deadline_in = self.next_piece_slack(
            now_s, encoder_now, tracks, skip, arm.free_at_s, speed_mps
        )
        self.last_slack = slack

        if self.stopped:
            self._advance_stop(arm, dt_s)
        elif slack < self.stop_slack and deadline_in < self.stop_horizon:
            self.stopped, self._stop_phase, self._idle_s = True, 1, 0.0  # a miss is imminent
            self.stop_count += 1
        if self.stopped:
            self.command_mps = self.v_lo
            return self.command_mps

        if slack == float("inf"):  # nothing waiting: drift back to the nominal speed
            self.integral *= max(0.0, 1.0 - 2.0 * dt_s)
            self.command_mps = self.v_nom
            return self.command_mps
        err = min(slack, self.clip) - self.setpoint
        raw = self.v_nom + self.kp * err + self.ki * self.integral
        out = min(max(raw, self.v_lo), self.v_hi)
        if out == raw:  # anti-windup: only accumulate while the output is not saturated
            self.integral = min(max(self.integral + err * dt_s, -self.i_max), self.i_max)
        self.command_mps = out
        return out

    def _advance_stop(self, arm, dt_s: float) -> None:
        """Hold the belt until the arm has finished its cycle AND gripped the waiting piece.

        Releasing the moment the arm frees up would let the belt accelerate before the scheduler
        has dispatched, and the planner would then see the piece carried past the zone. So: phase 1
        waits for the arm to finish; phase 2 waits for the next pick to reach its grip (or for
        nothing to be dispatched)."""
        from meatpick.control.arm_model import ArmState

        if self._stop_phase == 1:
            if not arm.busy:
                self._stop_phase, self._idle_s = 2, 0.0
            return
        if arm.state == ArmState.MOVING_TO_PICK:
            self._idle_s = 0.0  # a pick is on its way: keep the piece still
        elif arm.busy:
            self.stopped, self._stop_phase = False, 0  # gripped (or later): let the belt go
        else:
            self._idle_s += dt_s
            if self._idle_s > 0.2:  # nothing was dispatched: do not hold the belt for nothing
                self.stopped, self._stop_phase = False, 0

    # --- what the planner needs ---------------------------------------------------------
    def predict_travel(self, dt_s: float) -> float:
        """Belt travel in mm over the next ``dt_s`` seconds if the current command holds and the
        belt ramps at the acceleration limit."""
        v0 = self.speed_mps * 1000.0
        v1 = self.command_mps * 1000.0
        a = self.accel * 1000.0
        if dt_s <= 0:
            return 0.0
        t_ramp = abs(v1 - v0) / a
        sign = 1.0 if v1 >= v0 else -1.0
        if dt_s <= t_ramp:
            return v0 * dt_s + 0.5 * sign * a * dt_s * dt_s
        ramp = v0 * t_ramp + 0.5 * sign * a * t_ramp * t_ramp
        return ramp + v1 * (dt_s - t_ramp)
