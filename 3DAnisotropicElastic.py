import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


### 读取弹性矩阵
os.system('grep -A 8 "TOTAL ELASTIC" OUTCAR | tail -6 > ecs.dat')
df_ecs = pd.read_csv('ecs.dat', header=None, index_col=0, delimiter=r"\s+")
dat_ecs = df_ecs.values       ### 得到vasp计算出来的弹性矩阵


### 将vasp给出的弹性矩阵转换成满足Voigt下标的矩阵形式并将单位转换为GPa
dat_c = np.zeros(dat_ecs.shape)
for m in range(6):
    if m <= 2:
        m_ = m
    elif m == 3 or m == 4:
        m_ = m + 1
    elif m == 5:
        m_ = 3
    for n in range(6):
        if n <= 2:
            n_ = n
        elif n == 3 or n == 4:
            n_ = n + 1
        elif n == 5:
            n_ = 3
        dat_c[m, n] = dat_ecs[m_, n_]*0.1
print(dat_c)


### 计算柔度矩阵
dat_s = np.linalg.inv(dat_c)       

### 将二维柔度矩阵转换成四维柔度矩阵
def tranS_ijkl(dat_s):
    '''
    将6x6的二维柔度矩阵转换成3x3x3x3的柔度矩阵
    '''
    S_ijkl = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            if i==j:
                s_m = i
                ij_eq = True
            elif i+j == 3:
                s_m = 3
                ij_eq = False
            elif i+j == 2:
                s_m = 4
                ij_eq = False
            elif i+j == 1:
                s_m = 5
                ij_eq = False
            for k in range(3):
                for l in range(3):
                    if k==l:
                        s_n = k
                        kl_eq = True
                    elif k+l == 3:
                        s_n = 3
                        kl_eq = False
                    elif k+l == 2:
                        s_n = 4
                        kl_eq = False
                    elif k+l == 1:
                        s_n = 5
                        kl_eq = False
                    if ij_eq and kl_eq:
                        S_ijkl[i, j, k, l] = dat_s[s_m, s_n]
                    elif not ij_eq and not kl_eq:
                        S_ijkl[i, j, k, l] = dat_s[s_m, s_n]/4.0
                    else:
                        S_ijkl[i, j, k, l] = dat_s[s_m, s_n]/2.0
    return S_ijkl


S_ijkl = tranS_ijkl(dat_s)
sample_count = 500       ### 网格数量

### 在空间球面上均匀采样
v_theta = np.linspace(0, np.pi, sample_count)
v_phi = np.linspace(0, 2.0*np.pi, sample_count)
V_theta, V_phi = np.meshgrid(v_theta, v_phi)
a_x = np.sin(V_theta)*np.cos(V_phi)
a_y = np.sin(V_theta)*np.sin(V_phi)
a_z = np.cos(V_theta)
v_A = [a_x, a_y, a_z]

E_a_1 = np.zeros((sample_count, sample_count))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                E_a_1 = E_a_1 + v_A[i]*v_A[j]*v_A[k]*v_A[l]*S_ijkl[i, j, k, l]

E_a = 1.0/E_a_1     ### 取倒数得到杨氏模量，单位为GPa
plt.figure(dpi=200)
ax = plt.subplot(111, projection='3d')
ax.set_xlabel('Ex')
ax.set_ylabel('Ey') 
ax.set_zlabel('Ez') 
ax.set_title("Young's Modulus(GPa)")
ax.__eq__ = True
ax.plot_surface(E_a*a_x, E_a*a_y, E_a*a_z, rstride=1, cstride=1, color='y')
plt.savefig('young3D.jpg')
plt.show()
