import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

I = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
V = np.array([1.23, 1.38, 2.06, 2.47, 3.17])
y = V/I

# ## Estimating the Parameters
## Batch Solution
H = np.ones((5,2))
H[:, 0] = I
print(H)
x_ls = inv(H.T.dot(H)).dot(H.T.dot(V))
print('The parameters of the line fit are ([R, b]):')
print(x_ls)

#Plot
I_line = np.arange(0, 0.8, 0.1)
V_line = x_ls[0]*I_line + x_ls[1]

plt.scatter(I, V)
plt.plot(I_line, V_line)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()


## Recursive solution
#Initialize the 2x2 covaraince matrix
P_k = np.array([[10,0], [0, 0.2]])

#Initialize the parameter estimate x
x_k = np.array([4,0]).T

#Our measurement variance
Var = 0.0225

#Pre allocate our solutions so we can save the estimate at every step
num_meas = I.shape[0]
print(y)
x_hist = np.zeros((num_meas + 1,2))
P_hist = np.zeros((num_meas + 1,2,2))

x_hist[0] = x_k
P_hist[0] = P_k

#Iterate over the measurements
for k in range(num_meas):
    H_k = np.array([I[k], 1]).reshape(1,2)
    K_k = (P_hist[k].dot(H_k.T)).dot(inv(H_k.dot((P_hist[k]).dot(H_k.T)) + Var))                
    x_k = x_hist[k] + K_k.dot(V[k] - (H_k.dot(x_hist[k])))
    P_k = (np.eye(2) - (K_k.dot(H_k))).dot(P_hist[k])

    P_hist[k+1] = P_k
    x_hist[k+1] = x_k
    
print('The parameters of the line fit are ([R, b]):')
print(x_k)


#Plot
plt.scatter(I, V, label='Data')
plt.plot(I_line, V_line, label='Batch Solution')
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)

I_line = np.arange(0, 0.8, 0.1)
for k in range(num_meas):
    V_line = x_hist[k,0]*I_line + x_hist[k,1]
    plt.plot(I_line, V_line, label='Measurement {}'.format(k))

plt.legend()
plt.show()
