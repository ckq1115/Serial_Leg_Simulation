"""Planar centroidal momentum from CAD inertials and encoder/IMU samples.

No MuJoCo runtime state is used. XML provides the exported, fixed CAD inertials.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from math import atan2, hypot, sqrt
import numpy as np
from utils import PROJECT_ROOT


class JumpMomentumModel:
    PARTS = ('thigh', 'hip_knee_driver', 'upper_coupler', 'knee_node',
             'lower_coupler', 'shank', 'wheel')

    def __init__(self, left_leg, right_leg):
        self.legs = (left_leg, right_leg)
        xml = ET.parse(PROJECT_ROOT/'mjcf/bigeye_robot.xml').getroot()
        names = ['base']+[side+'_'+part+'_link' for side in ('left','right') for part in self.PARTS]
        rows=[]
        for name in names:
            inertial=xml.find(f".//body[@name='{name}']/inertial")
            pos=np.fromstring(inertial.get('pos'),sep=' ')
            inertia=np.fromstring(inertial.get('fullinertia'),sep=' ')
            rows.append((float(inertial.get('mass')),pos[[0,2]],inertia[1]))
        # Guide rollers belong to the chassis for the airborne momentum model.
        mass, center, pitch_inertia = rows[0]
        chassis = xml.find(".//body[@name='base']")
        guides = []
        for body in chassis.findall('body'):
            if body.get('name').startswith('guide_wheel_'):
                inertial = body.find('inertial')
                offset = np.fromstring(body.get('pos'), sep=' ')
                com = offset + np.fromstring(inertial.get('pos'), sep=' ')
                inertia = np.fromstring(inertial.get('fullinertia'), sep=' ')
                guides.append((float(inertial.get('mass')), com[[0, 2]], inertia[1]))
        if guides:
            chassis_parts = [(mass, center, pitch_inertia), *guides]
            total_mass = sum(m for m, _, _ in chassis_parts)
            center = sum(m*p for m, p, _ in chassis_parts)/total_mass
            pitch_inertia = sum(i + m*np.dot(p-center, p-center) for m, p, i in chassis_parts)
            rows[0] = (total_mass, center, pitch_inertia)
        self.mass=np.array([a[0] for a in rows])
        self.com=np.array([a[1] for a in rows])
        self.inertia=np.array([a[2] for a in rows])
        self._kinematics_cache = [None, None]

    def pose(self, side, front, back):
        leg=self.legs[side];p=leg.points;rot=leg.rotate
        B=rot(p['B'],front);P=rot(p['P'],back);K=rot(p['K'],back)
        dx, dz = B-P
        length = hypot(dx, dz)
        ux, uz = dx/length, dz/length
        a = (leg.pc**2-leg.bc**2+length**2)/(2*length)
        h = leg.branch*sqrt(max(0.,leg.pc**2-a*a))
        cx, cz = a*ux-h*uz, a*uz+h*ux
        pc = atan2(cz, cx)-leg.pc_angle
        bc = atan2(P[1]+cz-B[1], P[0]+cx-B[0])-atan2(*(p['C']-p['B'])[::-1])
        D=P+rot(p['D']-p['P'],pc);W=K+rot(p['W']-p['K'],pc)
        angles=np.array([back,front,bc,pc,back,pc,0.])
        origins = np.array([(0., 0.), (0., 0.), B, P, D, K, W])
        x, z = self.com[1+7*side:8+7*side].T
        cosines, sines = np.cos(angles), np.sin(angles)
        positions = origins + np.column_stack((cosines*x-sines*z, sines*x+cosines*z))
        # Wheel CoM offset is negligible, but its spin inertia is retained.
        return positions,angles,W

    def leg_kinematics(self, side, q):
        # Air control evaluates two velocity states at the same joint angles.
        key = tuple(q)
        cached = self._kinematics_cache[side]
        if cached is not None and cached[0] == key:
            return cached[1]
        positions, _, wheel = self.pose(side, *q)
        position_jacobian = []
        angle_jacobian = []
        for delta in (np.array([1e-5, 0.]), np.array([0., 1e-5])):
            p, a, _ = self.pose(side, *(q+delta))
            n, b, _ = self.pose(side, *(q-delta))
            position_jacobian.append((p-n)/2e-5)
            angle_jacobian.append(((a-b+np.pi)%(2*np.pi)-np.pi)/2e-5)
        result = positions, wheel, np.asarray(position_jacobian), np.asarray(angle_jacobian)
        self._kinematics_cache[side] = key, result
        return result

    def evaluate(self, state):
        phi=state['euler'][1];rate=state['gyro'][1]
        co,si=np.cos(phi),np.sin(phi);R=np.array([[co,si],[-si,co]])
        positions=[R@self.com[0]];velocities=[np.array([positions[0][1],-positions[0][0]])*rate]
        angular=[rate];wheels=[]
        for side,name in enumerate(('left','right')):
            q=np.array([state[name+'_front_pos'],state[name+'_back_pos']])
            dq=np.array([state[name+'_front_vel'],state[name+'_back_vel']])
            r, W, position_jacobian, angle_jacobian = self.leg_kinematics(side, q)
            v = position_jacobian[0]*dq[0] + position_jacobian[1]*dq[1]
            ad = angle_jacobian[0]*dq[0] + angle_jacobian[1]*dq[1]
            r=r@R.T;v=v@R.T+np.column_stack((r[:,1],-r[:,0]))*rate
            omega=rate-ad
            omega[-1]=rate-ad[5]+state[name+'_wheel_vel']
            positions.extend(r);velocities.extend(v);angular.extend(omega);wheels.append(R@W)
        positions=np.array(positions);velocities=np.array(velocities)
        center=np.average(positions,axis=0,weights=self.mass)
        speed=np.average(velocities,axis=0,weights=self.mass)
        r=positions-center;v=velocities-speed
        H=np.sum(self.inertia*np.array(angular)+self.mass*(r[:,1]*v[:,0]-r[:,0]*v[:,1]))
        return float(H),center,np.mean(wheels,axis=0)
