"""Write one JSONL trace per self-righting episode, without feeding the controller."""
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
from time import monotonic
from uuid import uuid4


class RecoveryLog:
    def __init__(self, directory, model, wheel_geom_id):
        self.directory = Path(directory)
        root = model.body_rootid[model.geom_bodyid[wheel_geom_id]]
        self.robot_geoms = model.body_rootid[model.geom_bodyid] == root
        self.pretrigger = deque(maxlen=26)
        self.last_pretrigger = float('-inf')
        self.event_count = 0
        self.file = None
        self.disabled = False
        self.recovered_at = None
        self.last_flush = 0.
        self.wall_start = monotonic()

    def write(self, row):
        self.file.write(json.dumps(row, default=lambda value: value.tolist(), separators=(',', ':'))+'\n')

    def finish(self, reason):
        if self.file is None:
            return
        try:
            self.write({'type': 'end', 'reason': reason, 'recovered_at_s': self.recovered_at})
            self.file.close()
        except OSError as exc:
            self.fail(exc)
        finally:
            self.file = None
            self.recovered_at = None

    def reset(self):
        self.finish('reset')
        self.pretrigger.clear()
        self.last_pretrigger = float('-inf')
        self.event_count = 0

    def wants_idle(self, now):
        return not self.disabled and (self.file is not None or now-self.last_pretrigger >= .02-1e-9)

    def fail(self, exc):
        if self.file is not None:
            try:
                self.file.close()
            except OSError:
                pass
        self.file = None
        self.disabled = True
        print(f'Recovery log disabled: {exc}', file=sys.stderr)

    def record(self, controller, sensors, x_lqr, target, forces, balance_ctrl, recovery_ctrl, ctrl):
        if self.disabled:
            return
        r, model, data = controller.recovery, controller.model, controller.data
        now = float(data.time)
        events = r.events[self.event_count:]
        self.event_count = len(r.events)
        if r.active and self.recovered_at is not None:
            self.finish('recovered')
        if self.disabled:
            return
        if self.file is None and not r.active and now-self.last_pretrigger < .02-1e-9:
            return

        pairs = {}
        for contact in data.contact:
            a, b = contact.geom1, contact.geom2
            if a < 0 or b < 0 or not (self.robot_geoms[a] or self.robot_geoms[b]):
                continue
            pair = tuple(sorted((a,b)))
            count, distance = pairs.get(pair, (0, contact.dist))
            pairs[pair] = (count+1, min(distance, contact.dist))
        contacts = [[model.geom(a).name or a, model.geom(b).name or b, count, distance]
                    for (a,b),(count,distance) in pairs.items()]
        row = {
            'type': 'sample', 't_s': now, 'wall_elapsed_s': monotonic()-self.wall_start,
            'active': r.active, 'phase': r.phase,
            'phase_elapsed_s': r.elapsed, 'retries': r.retries, 'reason': r.block_reason,
            'routes': r.routes.copy(), 'leg_done': r.leg_done.copy(), 'blend': r.blend, 'handoff': r.handoff,
            'support_action': r.support_action, 'support_alignment': r.support_action == 'align',
            'decisions': r.trace, 'sensors': sensors, 'jump_state': controller.jump_state,
            'contact_estimate': controller.gc.contact_state.copy(),
            'estimated_normal_force_n': [controller.gc.fn_left, controller.gc.fn_right],
            'slip': controller.gc.slip_active.copy(), 'stuck': controller.gc.stuck_active.copy(),
            'lqr_state': x_lqr.ravel().tolist(), 'lqr_target': target.ravel().tolist(),
            'target_leg_length_m': controller.target_leg_length, 'lqr_output': controller.U_lqr.ravel().tolist(),
            'balance_axial_force_n': None if forces is None else list(forces),
            'balance_ctrl_nm': None if balance_ctrl is None else balance_ctrl.tolist(),
            'recovery_ctrl_nm': None if recovery_ctrl is None else recovery_ctrl.tolist(),
            'ctrl_nm': ctrl.tolist(), 'actuator_force_previous_nm': data.actuator_force.tolist(),
            'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(), 'sim_contacts': contacts,
            'external_body_wrenches': [[i, data.xfrc_applied[i].tolist()]
                                       for i in data.xfrc_applied.any(axis=1).nonzero()[0].tolist()],
            'sim_warnings': [w.number for w in data.warning],
        }
        try:
            if self.file is None:
                if not r.active:
                    self.pretrigger.append(row)
                    self.last_pretrigger = now
                    return
                self.directory.mkdir(parents=True, exist_ok=True)
                name = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]+'.jsonl'
                self.file = (self.directory/name).open('x', encoding='utf8')
                self.write({'type': 'start', 'version': 1, 't_s': now,
                            'model': model.names.split(b'\0', 1)[0].decode('utf8', errors='replace'),
                            'control_period_s': controller.dt, 'recovery_config': r.cfg,
                            'actuators': [model.actuator(i).name for i in range(model.nu)],
                            'torque_limits_nm': [r.hip_limit]*4+[r.wheel_limit]*2})
                for frame in self.pretrigger:
                    self.write(frame)
                self.pretrigger.clear()
            for event in events:
                self.write({'type': 'transition', 't_s': now, **event})
            self.write(row)
            if not r.active and self.recovered_at is None:
                self.recovered_at = now
            if self.recovered_at is not None and now-self.recovered_at >= .3:
                self.finish('recovered')
            elif events or monotonic()-self.last_flush >= .25:
                self.file.flush()
                self.last_flush = monotonic()
        except OSError as exc:
            self.fail(exc)
