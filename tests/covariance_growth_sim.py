import numpy as np
import matplotlib.pyplot as plt

x=[]
sigma_x = 50*1000   # [m]
sigma_dx = 0.01     # [m/s]
sigma_dxm = 0.001   # [m/s]
sigma_dx = np.sqrt(sigma_dx**2 + sigma_dxm**2)

T = 60*60*24*3 # [s]
for t in range(T):
    x.append(sigma_x*np.sqrt(1+((sigma_dx*t)/sigma_x)**2))

plt.figure(figsize=(8, 6))
plt.plot(range(T), x, label='x values', marker='o', linestyle='-', color='red')
plt.show()
