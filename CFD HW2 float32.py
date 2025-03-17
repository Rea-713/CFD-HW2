#%%
# Importing Modules
import numpy as np
from matplotlib import pyplot as plt

#%%
# Exact Functions and Parameters
def u(k, x):
    u = np.e**(-x)*np.sin(k*x).astype(np.float32)
    return u

def dudx_exact(k, x):
    dudx = np.e**(-x) * (k*np.cos(k*x) .astype(np.float32)- np.sin(k*x).astype(np.float32))
    return dudx

def dudx2_exact(k, x):
    dudx2 = np.e**(-x) * ((1 - k**2) * np.sin(k*x).astype(np.float32) - 2*k*np.cos(k*x).astype(np.float32))
    return dudx2

dxs = np.logspace(-6, -1, 6)
dxs1 = dxs
dxs2 = dxs**2
dxs4 = dxs**4
ks = np.array([1, 2])                                                          # No. of waves
Lx = 2 * np.pi                                                                 # Computing Domain
flag1 = 1
flag2 = 1
flag3 = 1
flag4 = 1
TE_trunc = []
TE_round = []
TE_abs = []
epsilon2 = np.finfo(np.float32).eps                                             #1e-16: Double precision machine epsilon, np.finfo(np.float32).eps: Single precision machine epsilon   

#%%
# ∂u/∂x
# 1st-order Central Difference
def central_diff1_2(dx, k):                                                    #1: Order of the function; 2: Order of the Accurary
    grid = np.arange(-2*dx, 3*dx, dx, dtype=np.float32)
    i = 2
    u = np.e**(-grid).astype(np.float32)*np.sin(k*grid).astype(np.float32)
    du = (u[i + 1] - u[i - 1]) / (2 * dx) # + O(dx^2)
    return du

print('1st-order Accurate Central Difference for ∂u/∂x')

TE_t11 = []
TE_r11 = []
TE_a11 = []
dxs0 = list(dxs)
for dx in dxs0:
    for k in ks:
        res = central_diff1_2(dx, k)
        exact = dudx_exact(k, 0)
        TEabs = exact - res
        TEt11 = ((dx)**2 / 6) * k * (k**2 - 3)
        TEr11 = epsilon2 / (2*dx)
        TE_t11.append(TEt11)
        TE_r11.append(TEr11)
        TE_a11.append(TEabs)
        print(f'No{flag1}: Numerial={res:.10f}, Exact={exact},  TE_t={TEt11:.0e}, TE_round={TEr11:.0e}, TE_abs={TEabs:.0e}, dx={dx:.0e}, k={k}')
        flag1 += 1
TE_trunc.append(TE_t11)
TE_round.append(TE_r11)
TE_abs.append(TE_a11)

# %%
# ∂u/∂x
# 2nd-order Accurate Central Difference

def central_diff1_4(dx, k):                                                    #1: Order of the function; 4: Order of the Accurary         
    grid = np.arange(-2*dx, 3*dx, dx, dtype=np.float32)
    i = 2
    u = np.e**(-grid).astype(np.float32)*np.sin(k*grid).astype(np.float32)
    du = (-u[i + 2] + 8 * u[i + 1] - 8 * u[i-1] + u[i - 2]) / (12 * dx)        # + O(dx^4)
    return du

print('')
print('2nd-order Accurate Central Difference for ∂u/∂x')

TE_t12 = []
TE_r12 = []
TE_a12 = []
for dx in dxs0:
    for k in ks:
        res = central_diff1_4(dx, k)
        exact = dudx_exact(k, 0)
        TEabs = exact - res
        TEt12 = (dx**4 / 30) * k * (k**4 - 10*k**2 + 5)
        TEr12 = 1.5 * epsilon2 / dx
        TE_t12.append(TEt12)
        TE_r12.append(TEr12)
        TE_a12.append(TEabs)
        print((f'No{flag2}: Numerial={res:.30f}, Exact={exact}, TE_t={TEt11:.0e}, TE_r={TEr12:.0e}, TE_abs={TEabs:.0e}, dx={dx:.0e}, k={k}'))
        flag2 += 1
        
TE_trunc.append(TE_t12)
TE_round.append(TE_r12)
TE_abs.append(TE_a12)

# %%
# ∂2u/∂x2
# 2th-order Accurate Central Difference

def central_diff2_2(dx, k):
    grid = np.arange(-2*dx, 3*dx, dx, dtype=np.float32)
    j = 2
    u = np.e**(-grid).astype(np.float32)*np.sin(k*grid).astype(np.float32)
    du = (u[j + 1] - 2 * u[j] + u[j - 1]) / (dx)**2                            # + O(dx^2)
    return du

print('')
print('2nd-order Accurate Central Difference for ∂2u/∂x2')

TE_t21 = []
TE_r21 = []
TE_a21 = []
for dx in dxs0:
    for k in ks:
        res = central_diff2_2(dx, k)
        exact = dudx2_exact(k, 0)
        TEabs = exact - res
        TEt21 = (dx**2 / 3) * k * (k**2 - 1)
        TEr21 = 4 * epsilon2 / dx**2
        TE_t21.append(TEt21)
        TE_r21.append(TEr21)
        TE_a21.append(TEabs)
        print((f'No{flag3}: Numerial={res:.30f}, Exact={exact}, TE_t={TEt21:.0e}, TE_r={TEr21:.0e}, TE_abs={TEabs:.0e}, dx={dx:.0e}, k={k}'))
        flag3 += 1
        
TE_trunc.append(TE_t21)
TE_round.append(TE_r21)
TE_abs.append(TE_a21)

# %%

# ∂2u/∂x2
# 1st-order Accurate Forward Difference

def forward_diff2_2(dx, k):
    grid = np.arange(0, 3*dx, dx, dtype=np.float32)
    j = 0
    u = np.e**(-grid) .astype(np.float32)* np.sin(k*grid).astype(np.float32)
    du = (u[j + 2] - 2*u[j + 1] + u[j]) / dx**2
    return du

print('')
print('1st-order Accurate Forward Difference for ∂2u/∂x2')

TE_t22 = []
TE_r22 = []
TE_a22 = []
for dx in dxs0:
    for k in ks:
        res = forward_diff2_2(dx, k)
        exact = dudx2_exact(k, 0)
        TEabs = exact - res
        TEt22 = -dx * k * (k**2 - 3)
        TEr22 = 4 * epsilon2 / dx**2
        TE_t22.append(TEt22)
        TE_r22.append(TEr22)
        TE_a22.append(TEabs)
        print((f'No{flag4}: Numerial={res:.30f}, Exact={exact}, TE_t={TEt22:.0e}, TE_r={TEr22:.0e}, TE_abs={TEabs:.0e}, dx={dx:.0e}, k={k}'))
        flag4 += 1
        
TE_trunc.append(TE_t22)
TE_round.append(TE_r22)
TE_abs.append(TE_a22)

# %%

l = len(dxs)*len(ks)
even = [j for j in range(0, l+1, 2)]
odd = [j for j in range(1, l+1, 2)]

# %%

# Figure Plotting for ∂u/∂x 1
fig1 = plt.figure(figsize=(25, 4))
ax1 = fig1.add_subplot(1, 3, 1)

TER_list = []
TET_list = []
TEA_list = []

for e1 in even:
    if len(TER_list) == len(dxs):
        break
    TER_list.append(abs(TE_round[0][e1]))
    TET_list.append(abs(TE_trunc[0][e1]))
    TEA_list.append(abs(TE_abs[0][e1]))
    
plt.plot(dxs, TER_list, marker = 'o')
ax1.set(xscale = 'log', yscale = 'log' ,
        title = 'Round-off Error Anlysis of 1st-order Accurate Central Difference for ∂u/∂x (k = 1)', 
        xlabel = 'dx', ylabel = 'TE_round')
ax1.grid(True)

ax2 = fig1.add_subplot(1, 3, 2)
plt.plot(dxs, TET_list, marker = 's')
ax2.set(xscale = 'log', yscale = 'log',
        title = 'Trunc Error Anlysis of 1st-order Accurate Central Difference for ∂u/∂x (k = 1)', 
        xlabel = 'dx', ylabel = 'TE_trunc')
ax2.grid(True)

ax3 = fig1.add_subplot(1, 3, 3)
plt.plot(dxs, TEA_list, marker = '^')
ax3.set(xscale = 'log', yscale = 'log' ,
        title = 'Absolute Error Anlysis of 1st-order Accurate Central Difference for ∂u/∂x (k = 1)', 
        xlabel = 'dx', ylabel = 'TE_abs')
ax3.grid(True)

plt.tight_layout()

# %%

# Figure Plotting for ∂u/∂x 2

fig2 = plt.figure(figsize=(25, 4))
ax1 = fig2.add_subplot(1, 3, 1)

TER_list = []
TET_list = []
TEA_list = []

for o1 in odd:
    if len(TER_list) == len(dxs):
        break
    TER_list.append(TE_round[1][o1])
    TET_list.append(abs(TE_trunc[1][o1]))
    TEA_list.append(abs(TE_abs[1][o1]))
    
    
plt.plot(dxs, TER_list, marker = 'o')
ax1.set(xscale = 'log', yscale = 'log',
        title = 'Round-off Error Anlysis of 2nd-order Accurate Central Difference for ∂u/∂x (k = 2)', 
        xlabel = 'dx', ylabel = 'TE_round')
ax1.grid(True)

ax2 = fig2.add_subplot(1, 3, 2)
plt.plot(dxs, TET_list, marker = 's')
ax2.set(xscale = 'log', yscale = 'log',
        title = 'Trunc Error Anlysis of 2nd-order Accurate Central Difference for ∂u/∂x (k = 2)', 
        xlabel = 'dx', ylabel = 'TE_trunc')
ax2.grid(True)

ax3 = fig2.add_subplot(1, 3, 3)
plt.plot(dxs, TEA_list, marker = '^')
ax3.set(xscale = 'log', yscale = 'log',
        title = 'Absolute Error Anlysis of 2nd-order Accurate Central Difference for ∂u/∂x (k = 2)', 
        xlabel = 'dx', ylabel = 'TE_abs')
ax3.grid(True)

plt.tight_layout()

# %%

# Figure Plotting for ∂2u/∂x2 1
fig3 = plt.figure(figsize=(25, 4))
ax1 = fig3.add_subplot(1, 3, 1)

TER_list = []
TET_list = []
TEA_list = []

for e1 in even:
    if len(TER_list) == len(dxs):
        break
    TER_list.append(abs(TE_round[2][e1]))
    TET_list.append(abs(TE_trunc[2][e1]))
    TEA_list.append(abs(TE_abs[2][e1]))
    
plt.plot(dxs, TER_list, marker = 'o')
ax1.set(xscale = 'log', yscale = 'log',
        title = 'Round-off Error Anlysis of 1st-order Accurate Central Difference for ∂2u/∂x2 (k = 1)', 
        xlabel = 'dx', ylabel = 'TE_round')
ax1.grid(True)

ax2 = fig3.add_subplot(1, 3, 2)
plt.plot(dxs, TET_list, marker = 's')
ax2.set(title = 'Trunc Error Anlysis of 1st-order Accurate Central Difference for ∂2u/∂x2 (k = 1)', 
        xlabel = 'dx', ylabel = 'TE_trunc')
ax2.grid(True)

ax3 = fig3.add_subplot(1, 3, 3)
plt.plot(dxs, TEA_list, marker = '^')
ax3.set(xscale = 'log', yscale = 'log',
        title = 'Absolute Error Anlysis of 1st-order Accurate Central Difference for ∂2u/∂x2 (k = 1)', 
        xlabel = 'dx', ylabel = 'TE_abs')
ax3.grid(True)

plt.tight_layout()

# %%

# Figure Plotting for ∂2u/∂x2 2
fig4 = plt.figure(figsize=(25, 4))
ax1 = fig4.add_subplot(1, 3, 1)

TER_list = []
TET_list = []
TEA_list = []

for o1 in odd:
    if len(TER_list) == len(dxs):
        break
    TER_list.append(abs(TE_round[-1][o1]))
    TET_list.append(abs(TE_trunc[-1][o1]))
    TEA_list.append(abs(TE_abs[-1][o1]))
    
plt.plot(dxs, TER_list, marker = 'o')
ax1.set(xscale = 'log', yscale = 'log',
        title = 'Round-off Error Anlysis of 2nd-order Accurate Central Difference for ∂2u/∂x2 (k = 2)', 
        xlabel = 'dx', ylabel = 'TE_round')
ax1.grid(True)

ax2 = fig4.add_subplot(1, 3, 2)
plt.plot(dxs, TET_list, marker = 's')
ax2.set(xscale = 'log', yscale = 'log',
        title = 'Trunc Error Anlysis of 1st-order Accurate Forward Difference for ∂2u/∂x2 (k = 2)', 
        xlabel = 'dx', ylabel = 'TE_trunc')
ax2.grid(True)

ax3 = fig4.add_subplot(1, 3, 3)
plt.plot(dxs, TEA_list, marker = '^')
ax3.set(xscale = 'log', yscale = 'log',
        title = 'Absolute Error Anlysis of 1st-order Accurate Central Difference for ∂u/∂x (k = 2)', 
        xlabel = 'dx', ylabel = 'TE_abs')
ax3.grid(True)

plt.tight_layout()

# %%

plt.show()
