import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]
print(b[1, 0])

v_var = 0.01  # translation velocity variance  
om_var = 15  # rotational velocity variance 
r_var = 0.001  # range measurements variance
b_var = 0.001 # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance

# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

 
# ## Correction Step
import math

def measurement_update(lk, rk, bk, P_check, x_check):
    
    xi = x_check[0, 0]
    yi = x_check[1, 0]
    thi = x_check[2, 0]
    d = 0
    
    # 1. Compute measurement Jacobian
    deno = (lk[0] - xi)**2 + (lk[1] - yi)**2
    H_km = np.zeros([2, 3])
    H_km[0, 0] = (xi - lk[0])/np.sqrt(deno)
    H_km[0, 1] = (yi - lk[1])/np.sqrt(deno)
    H_km[0, 2] = 0
    H_km[1, 0] = -(yi - lk[1])/(deno)
    H_km[1, 1] = (xi - lk[0])/(deno)
    H_km[1, 2] = -1
    M_km = np.mat([[1, 0],[0, 1]])
    
    # 2. Compute Kalman Gain
    kki = (H_km@(P_check))@(H_km.T) + (M_km@(cov_y))@(M_km.T)
    kkinv = np.linalg.inv(kki)
    Kk = (P_check@(H_km.T))@(kkinv)
    
    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    ykl = np.mat([[rk], [bk]])
    ykcheck = np.zeros([2,1])
    ykcheck[0] = np.sqrt((lk[0] - xi - d*np.cos(thi))**2 + (lk[1] - yi - d*np.sin(thi))**2)
    ykcheck[1] = wraptopi(np.arctan2(lk[1] - yi - d*np.sin(thi), lk[0] - xi - d*np.cos(thi)) - thi)
    #print(Kk)
    
    # 4. Correct covariance
    x_check = x_check + (Kk@(ykl - ykcheck))
    x_check[2, 0] = wraptopi(x_check[2, 0])
    iden = np.mat([[1,0,0], [0,1,0], [0,0,1]])
    P_check = (iden - Kk@(H_km))@(P_check)
    return x_check, P_check


# ## Prediction Step
#### 5. Main Filter Loop #######################################################################
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    x_check = np.mat([[x_est[k-1][0]], [x_est[k-1][1]], [wraptopi(x_est[k-1][2])]])

    # 2. Motion model jacobian with respect to last state
    F_km = np.zeros([3, 3])
    F_km[0,0] = 1
    F_km[0,1] = 0
    F_km[0,2] = v[k-1]*np.sin(x_check[2, 0])
    F_km[1,0] = 0
    F_km[1,1] = 1
    F_km[1,2] = -v[k-1]*np.cos(x_check[2, 0])
    F_km[2,0] = 0
    F_km[2,1] = 0
    F_km[2,2] = 1
    
    # 3. Motion model jacobian with respect to noise
    L_km = np.zeros([3, 2])
    L_km[0,0] = delta_t*np.cos(x_check[2, 0])
    L_km[0,1] = 0
    L_km[1,0] = delta_t*np.sin(x_check[2, 0])
    L_km[1,1] = 0
    L_km[2,0] = 0
    L_km[2,1] = delta_t

    
    # 4. Propagate uncertainty
    G_km = L_km
    U_km = np.mat([[v[k-1]], [om[k-1]]])
    x_check = x_check + G_km@(U_km)
    x_check[2, 0] = wraptopi(x_check[2, 0])
    P_check = (F_km@(P_est[k-1]))@(F_km.T) + (L_km@(Q_km))@(L_km.T)
    
    # 5. Update state estimate
    for i in range(len(r[k])):
        b[k, i] = wraptopi(b[k, i])
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)
    
    # Set final state predictions
    x_est[k, 0] = x_check[0, 0]
    x_est[k, 1] = x_check[1, 0]
    x_est[k, 2] = x_check[2, 0]
    P_est[k, :, :] = P_check

# Let's plot the resulting state estimates:

# In[14]:


e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()



