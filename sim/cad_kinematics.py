"""Six-bar CAD geometry in the controller's native output-joint coordinates."""
import json
from math import atan2, cos, hypot, sin, sqrt
from pathlib import Path
import numpy as np
from kinematics import FiveLinkLeg

class CadSixBarLeg(FiveLinkLeg):
    def __init__(self,side,link_lengths,dt):
        super().__init__(side,link_lengths,dt)
        p=json.loads((Path(__file__).resolve().parents[1]/'config/cad_recognition.json').read_text(encoding='utf8'))
        self.points={k:np.array(v)[[0,2]]*.001 for k,v in p['legs'][side]['pivots_B_mm'].items()}
        self.points={k:v-self.points['A'] for k,v in self.points.items()}
        p=self.points;self.pc=np.linalg.norm(p['C']-p['P']);self.bc=np.linalg.norm(p['C']-p['B'])
        self.branch=np.sign(np.linalg.det(np.stack([p['B']-p['P'],p['C']-p['P']])))
        self.pc_angle=np.arctan2(*(p['C']-p['P'])[::-1])
        self._wheel_from_knee = self.points['W']-self.points['K']
        self._wheel_offset_angle = atan2(self._wheel_from_knee[1], self._wheel_from_knee[0])
        self._spring_thigh_arm = p['S'] - p['K']
        self._spring_shank_arm = p['T'] - p['K']
        spring_span = self._spring_thigh_arm - self._spring_shank_arm
        self._spring_zero_length = hypot(*spring_span)
        self.theta_body=0.0
        self.theta_world=0.0
        self.shank_rate=0.0
        self._shank_last=None
    @staticmethod
    def rotate(p,q):
        c,s=cos(q),sin(q);return np.array([c*p[0]-s*p[1],s*p[0]+c*p[1]])
    def _wheel_geometry(self, front, back):
        p = self.points
        cf, sf = cos(front), sin(front)
        cb, sb = cos(back), sin(back)
        bx, bz = p['B']
        px, pz = p['P']
        dx = (cf*bx - sf*bz) - (cb*px - sb*pz)
        dz = (sf*bx + cf*bz) - (sb*px + cb*pz)
        distance = hypot(dx, dz)
        ux, uz = dx/distance, dz/distance
        along = (self.pc**2-self.bc**2+distance**2)/(2*distance)
        height = self.branch*sqrt(max(0., self.pc**2-along**2))
        angle = atan2(along*uz+height*ux, along*ux-height*uz)-self.pc_angle
        kx, kz = p['K']
        wx, wz = self._wheel_from_knee
        ca, sa = cos(angle), sin(angle)
        return cb*kx-sb*kz+ca*wx-sa*wz, sb*kx+cb*kz+sa*wx+ca*wz, angle

    def _gas_spring_length(self, front, back):
        _, _, shank = self._wheel_geometry(front, back)
        span = self.rotate(self._spring_thigh_arm, back) - self.rotate(self._spring_shank_arm, shank)
        return hypot(*span)

    def gas_spring_torque(self, front, back, stiffness, reference_force):
        """按虚功计算气弹簧作用在前后髋关节上的有符号力矩。"""
        length = self._gas_spring_length(front, back)
        force = reference_force - stiffness * (length - self._spring_zero_length)
        # 两个偏导分别固定另一个髋角，沿六连杆闭环求弹簧长度变化。
        delta = 1e-6
        front_span = (self._gas_spring_length(front + delta, back)
                      - self._gas_spring_length(front - delta, back))
        back_span = (self._gas_spring_length(front, back + delta)
                     - self._gas_spring_length(front, back - delta))
        return force * front_span / (2 * delta), force * back_span / (2 * delta)

    def wheel_position(self, front, back):
        x, z, _ = self._wheel_geometry(front, back)
        return np.array([x, z])

    def _forward_raw(self,front,back):
        x, z, _ = self._wheel_geometry(front, back)
        return hypot(x, z), -atan2(x, -z)
    def reset_dynamic_state(self):
        super().reset_dynamic_state()
        self.shank_rate=0.0
        self._shank_last=None
    def forward(self,front,back,body_pitch):
        # Signed sagittal angle from world vertical. Positive: hip ahead of wheel.
        x, z, angle = self._wheel_geometry(front, back)
        self.length, self.theta_body = hypot(x, z), -atan2(x, -z)
        self.theta_world=self.theta_body+body_pitch
        self.theta=self.theta_world
        # Positive CAD joint rotation is about -Y; wheel spin is about +Y.
        shank = body_pitch - (angle + self._wheel_offset_angle)
        if self._shank_last is not None:
            delta=(shank-self._shank_last+np.pi)%(2*np.pi)-np.pi
            self.shank_rate=delta/self.length_dot.dt
        self._shank_last=shank
        self.length_dot.update(self.length);self.theta_dot.update(self.theta)
        self.length_ddot.update(self.length);self.theta_ddot.update(self.theta)
        return self.length,self.theta
    def jacobian(self,front,back):
        # At fixed body attitude d(theta_world)/dq == d(theta_body)/dq.
        eps=1e-6;result=np.empty((2,2))
        for i in range(2):
            q=np.array([front,back],dtype=float);q[i]+=eps;a=np.array(self._forward_raw(*q));q[i]-=2*eps;b=np.array(self._forward_raw(*q))
            result[:,i]=(a-b)/(2*eps)
        return result
    def vmc(self,force,torque,front,back):
        self.force=force
        self.front_torque,self.back_torque=self.jacobian(front,back).T@np.array([force,torque])
        return self.front_torque,self.back_torque
