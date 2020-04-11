import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Total population, N.
# N = 1000
N = 64.8
# Initial number of infected and recovered individuals, I0 and R0.
# I0, R0 = 1, 0
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
# beta, gamma = 0.2, 1./10 
beta, gamma = 0.594, 0.07 
# A grid of time points (in days)
t = np.linspace(0, 60, 60)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

for z,j,k,l in zip(t, S, I, R):
    print("{0}\t {1}\t {2}\t {3}\t".format(t,j/1000,k/1000,l/1000))

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
#fig.suptitle('This is a somewhat long figure title', fontsize=16)
ax = fig.add_subplot(111,  axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number of People (millions)')
ax.set_title('SIR Python Model Graph for Kalangala')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

# pieLabels = 'Susceptible', 'Infected', 'Recovered with immunity'
# pieData =[S/1000, I/1000, R/1000]
# figureObject, axesObject = plt.subplots()
# axesObject.pie(pieData,

#         labels=pieLabels,

#         autopct='%1.2f',
# )
# axesObject.axis('equal')
# plt.show()
