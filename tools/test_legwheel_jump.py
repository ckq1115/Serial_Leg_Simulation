"""Run legwheel's actual jump logic headless to measure pitch during jump."""
import sys; sys.path.insert(0, '../legwheel')
import numpy as np
import mujoco as mj
import time
from math import cos, sin

from user_class import RobotSensor, Leg, State, Change_length_fit
from mymath import PID_control
from kalman import Kalman

model = mj.MjModel.from_xml_path('../legwheel/legwheel.xml')
data = mj.MjData(model)
dt = model.opt.timestep

sensor = RobotSensor(model, data)
robot_state = State()
road = Kalman()
left = Leg('left', dt)
right = Leg('right', dt)

left_length_pos = PID_control(kp=2000, ki=0.0, kd=0, targ_value=0)
right_length_pos = PID_control(kp=2000, ki=0.0, kd=0, targ_value=0)
left_length_vel = PID_control(kp=100, ki=0, kd=0, targ_value=0)
right_length_vel = PID_control(kp=100, ki=0, kd=0, targ_value=0)
roll_pid = PID_control(kp=2000, ki=0, kd=1000, targ_value=0)

x_target=0; w_target=0; x_dot_target=0; w_dot_target=0
leg_length=0.12; G_m=104
jump=0; flag=0

k_coeff=np.array([[[6.2174,-11.2883,8.3893,-3.4140,0.7898], [1.4340,-1.8905,1.1834,-0.5364,-0.4896], [923.8710,-1092.6,502.6034,-110.6934,-57.0662], [364.5066,-420.9464,197.7163,-61.6707,29.2380], [347.2482,-494.2305,293.6293,-84.8201,-2.8636], [172.8698,-205.0662,97.0404,-23.7418,3.2421], [-2.1572,2.0367,-0.4234,-0.3492,-0.5612], [39.0661,-46.6079,21.7275,-4.8873,-2.8456], [16.1460,-16.5183,6.9372,-0.5876,0.6249], [13.1891,-10.1115,2.2276,-1.2081,-0.0720]],
                  [[6.2174,-11.2883,8.3893,-3.4140,0.7898], [-1.4340,1.8905,-1.1834,0.5364,0.4896], [923.8710,-1092.6,502.6034,-110.6934,-57.0662], [347.2482,-494.2305,293.6293,-84.8201,-2.8636], [364.5066,-420.9464,197.7163,-61.6707,29.2380], [172.8698,-205.0662,97.0404,-23.7418,3.2421], [2.1572,-2.0367,0.4234,0.3492,0.5612], [39.0661,-46.6079,21.7275,-4.8873,-2.8456], [13.1891,-10.1115,2.2276,-1.2081,-0.0720], [16.1460,-16.5183,6.9372,-0.5876,0.6249]],
                  [[17.2035,-25.3140,16.7064,-5.8122,-1.1309], [2.1946,-2.3090,0.7162,0.2570,-0.4818], [-1240.9,1624.2,-872.0751,248.8905,-39.5076], [2912.8,-2879.2,1171.0,-271.5191,-23.8006], [685.0781,-969.6809,519.8358,-124.8295,4.6944], [-243.6033,258.0985,-101.7795,17.4144,-6.2708], [2.2857,-2.4649,0.8261,0.2417,-0.5325], [-104.5821,129.5426,-64.9653,17.0979,-2.5537],[-71.1397,67.6541,-23.1166,-1.4280,-0.8085],[-65.3237,58.0976,-17.3916,-0.6455,-0.1529]],
                  [[17.2035,-25.3140,16.7064,-5.8122,-1.1309], [-2.1946,2.3090,-0.7162,-0.2570,0.4818], [-1240.9,1624.2,-872.0751,248.8905,-39.5076], [685.0781,-969.6809,519.8358,-124.8295,4.6944], [2912.8,-2879.2,1171.0,-271.5191,-23.8006], [-243.6033,258.0985,-101.7795,17.4144,-6.2708], [2.2857,-2.4649,0.8261,0.2417,-0.5325], [-104.5821,129.5426,-64.9653,17.0979,-2.5537],[-65.3237,58.0976,-17.3916,-0.6455,-0.1529],[-71.1397,67.6541,-23.1166,-1.4280,-0.8085]]])
theta_coeff=np.array([-90.3137,94.5175,-35.6182,3.7520,0.6523])
e1_coeff=np.array([7242.0, -7790.1, 3139.9, -571.7475, 41.9684])
e2_coeff=np.array([5.2420, -5.6453, 2.2743, -0.4113, 0.029])
e3_coeff=np.array([-233.8069, 251.4137, -101.2784, 18.4236, -1.3489])
change_length=Change_length_fit(np.zeros((4,10)),k_coeff,theta_coeff,e1_coeff,e2_coeff,e3_coeff)

length = {
    'left_front_big': 0.21, 'left_back_big': 0.21,
    'left_front_small': 0.2462962962962, 'left_back_small': 0.2462962962962,
    'left_wheel':0.05, 'right_front_big': 0.21, 'right_back_big': 0.21,
    'right_front_small': 0.2462962962962, 'right_back_small': 0.2462962962962,
    'right_wheel':0.055, 'R':0.165,
}

def step_ctrl(jump_in, flag_in):
    global jump, flag, leg_length
    jump=jump_in; flag=flag_in

    state = sensor.get_state()
    imu = {'acc': state['acc'], 'gyro': state['gyro'], 'euler': state['euler'], 'quat': state['quat']}
    motor = {
        'left_front_pos': state['joints']['motor_l1_pos'], 'left_back_pos': state['joints']['motor_l2_pos'],
        'left_wheel_pos': state['joints']['wheel_l_pos'], 'right_front_pos': state['joints']['motor_r1_pos'],
        'right_back_pos': state['joints']['motor_r2_pos'], 'right_wheel_pos': state['joints']['wheel_r_pos'],
        'left_front_vel': state['joints']['motor_l1_vel'], 'left_back_vel': state['joints']['motor_l2_vel'],
        'left_wheel_vel': state['joints']['wheel_l_vel'], 'right_front_vel': state['joints']['motor_r1_vel'],
        'right_back_vel': state['joints']['motor_r2_vel'], 'right_wheel_vel': state['joints']['wheel_r_vel'],
    }
    left.forward(motor, imu, length, robot_state.x[2,0])
    right.forward(motor, imu, length, robot_state.x[2,0])
    K=change_length.get_K((left.length+right.length)/2); F_c=K
    e2=change_length.get_e2((left.length+right.length)/2)
    e3_l=e3_r=change_length.get_e3((left.length+right.length)/2)
    left_length_pos.position_pid(leg_length,left.length)
    left_length_vel.position_pid(0,left.length_dot.Diff(left.length))
    right_length_pos.position_pid(leg_length,right.length)
    right_length_vel.position_pid(0,right.length_dot.Diff(right.length))
    roll_pid.position_pid(0,imu['euler'][0])

    max1=1500; max2=1000; max3=1000; jumped_r=0; jumped_l=0; G_m_local=104

    # legwheel jump logic
    if right.length<0.35 and jump and flag:
        jumped_r=5000; max1=0; max2=0
    elif left.length>=0.35 and jump:
        jumped_r=0; flag=0; max2=0; max3=0; G_m_local=0; leg_length=0.12
    elif left.length<=0.12 and jump:
        jumped_r=0; flag=0; max3=0; leg_length=0.12

    if left.length<0.35 and jump and flag:
        jumped_l=5000; max1=0; max2=0
    elif left.length>=0.35 and jump:
        jumped_l=0; flag=0; G_m_local=0; max2=0; max3=0; leg_length=0.12
    elif left.length<=0.12 and jump:
        jumped_r=0; flag=0; max3=0; leg_length=0.12

    if left_length_pos.output>max1: left_length_pos.output=max1
    if left_length_pos.output<-max1: left_length_pos.output=-max1
    if right_length_pos.output>max1: right_length_pos.output=max1
    if right_length_pos.output<-max1: right_length_pos.output=-max1
    if left_length_vel.output>max2: left_length_vel.output=max2
    if left_length_vel.output<-max2: left_length_vel.output=-max2
    if right_length_vel.output>max2: right_length_vel.output=max2
    if right_length_vel.output<-max2: right_length_vel.output=-max2
    if roll_pid.output>max3: roll_pid.output=max3
    if roll_pid.output<-max3: roll_pid.output=-max3

    z_body=sensor.body_acc(imu); road.filter(z_body,motor,length,dt)
    U=-F_c @ (robot_state.update_state(left, right, length, motor,imu, road)-robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target, 0-e2, 0-e3_l, 0-e3_r))

    F_l = left_length_vel.output + roll_pid.output + left_length_pos.output + G_m_local*cos(robot_state.x[3,0]) + jumped_l
    F_r = right_length_vel.output - roll_pid.output + right_length_pos.output + G_m_local*cos(robot_state.x[4,0]) + jumped_r

    left.vmc(F_l+12, U[0,0], motor, length); right.vmc(F_r+12, U[1,0], motor, length)
    U_out = np.block([[left.t_front],[right.t_front],[left.t_back],[right.t_back],[U[2,0]],[U[3,0]]])
    if abs(imu['euler'][0])>1.0 or abs(imu['euler'][1])>1.0:
        U_out[0:6,0]=0
    for i in range(6): data.ctrl[i] = U_out[i,0]
    mj.mj_step(model, data)
    return state, flag, left, right

# Initialize
mj.mj_forward(model, data)
for _ in range(500):
    step_ctrl(0, 0)  # normal control to stabilize

z0 = data.qpos[2]
state0 = sensor.get_state()
print(f'Stand: z={z0:.3f} p={np.degrees(state0["euler"][1]):.1f}deg L={0.5*(left.length+right.length):.3f}')

# First press: jump=1, flag=0
for _ in range(5): step_ctrl(1, 0)
# Release: jump=0, flag=1
for _ in range(5): step_ctrl(0, 1)
# Second press: jump=1, flag=1 → triggers!
print('Second press → trigger!')

peak_z = z0
for i in range(3000):
    state, flag_val, left_val, right_val = step_ctrl(1, 1)
    peak_z = max(peak_z, data.qpos[2])
    if i % 150 == 0:
        p = np.degrees(state['euler'][1])
        print(f'  t={data.time:.2f} L={0.5*(left.length+right.length):.4f} z={data.qpos[2]:.3f} p={p:.1f}deg flag={flag_val}')
    if not flag_val and i > 200:
        print(f'Jump phase complete')
        break

# Recovery
for _ in range(500):
    step_ctrl(0, 0)

state = sensor.get_state()
p = np.degrees(state['euler'][1])
print(f'Final: z={data.qpos[2]:.3f} jump_h={peak_z-z0:.3f}m p={p:.1f}deg')
