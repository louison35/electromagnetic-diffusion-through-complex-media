import matplotlib.pyplot as plt
import numpy as np
import random as rd
import cmath as cm

f = 7 * 10**9
omega = 2*np.pi*f
n = 6    #nombre de diffuseur
a = 0.1    #mètre
aa = 100      #centimètre
b = 0.4
bb = 400
c0 = 3 * 10**8

Ni = 1
fi = 0
A = True

while A:
    fi = (c0 * Ni) / (2 * a)
    if fi > f:
        A = False
    else:
        A = True
        Ni += 1

N = Ni - 1

k = [np.sqrt((omega / c0) ** 2 - ((i+1) * (np.pi / a)) ** 2) for i in range(N)]


def green_in(x1, y1, k1):
    g1 = np.zeros((N, len(x1)), dtype=complex)
    g2 = np.zeros((N, len(x1)), dtype=complex)

    for i in range(N):
        for j in range(len(x1)):
            g1[i, j] = (
                np.sqrt(2 / a)
                * ((-1) ** i)
                * (1 / np.sqrt(k1[i]))
                * np.sin(((i+1)*(np.pi/a))* y1[j])
                * np.exp(1j * k1[i] * x1[j])
            )

            g2[i, j] = (
                np.sqrt(2 / a)
                * ((-1) ** i)
                * (1 / np.sqrt(k1[i]))
                * np.sin(((i+1)*(np.pi/a)) * y1[j])
                * np.exp(1j * k1[i] * abs(b - x1[j]))
            )

    return np.concatenate((g1, g2))


def green_dd(x1, y1, k1):
    gi = np.zeros((len(x1), len(x1)), dtype=complex)
    g = np.zeros((len(x1), len(x1)), dtype=complex)
    for o in range(N):
        for i in range(len(x1)):
            for j in range(len(x1)):
                gi[i, j] = (
                    -np.sin(((o+1)*(np.pi/a)) * y1[i])
                    * np.sin(((o+1)*(np.pi/a)) * y1[j])
                    * (2 *(1/a) *( 1/k1[o]))
                    * np.exp(1j * k1[o] * abs(x1[j] - x1[i]))
                )
        g = g + gi
    return g


def S_matrix(x1, y1, k1, alpha):
    G1 = green_in(x1, y1, k1)
    G_t1 = np.transpose(G1)
    G_dd1 = green_dd(x1, y1, k1)
    w1 = alpha - G_dd1
    w_inv1 = np.linalg.inv(w1)
    S1 = np.dot(G1, np.dot(w_inv1, G_t1))
    return S1


def M_alpha(a_tot):
    alphai = np.zeros((len(a_tot), len(a_tot)), dtype=complex)
    for i in range(len(a_tot)):
        alphai[i, i] = 1 / (-a_tot[i] * 1j)
    result = alphai
    return result


S0 = np.zeros((2 * N, 2 * N), dtype=complex)
G0 = [np.exp(1j * b * k[o]) for o in range(N)]
t0 = np.diag(G0)
S0[:N, N : 2 * N] = t0
S0[N : 2 * N, :N] = t0


x = [0.257, 0.177, 0.08, 0.357, 0.105, 0.196]
y = [0.091, 0.04, 0.093, 0.074, 0.033, 0.048]
print("x =",x)
print("y =",y)


alpha0 = 12.5
alphal = [alpha0 for i in range(len(x))]
print("polarisabilité = ",alphal)

S = S_matrix(x, y, k, M_alpha(alphal))




Sf = S - S0
S_test = (np.dot(Sf, np.conj(np.transpose(Sf)))).copy()

plt.rcParams['figure.dpi'] = 300

plt.imshow(
    abs(Sf),
    vmin=0,
    vmax=1,
    extent=[1, 2 * N, 2 * N, 1],
)
plt.colorbar()
plt.show()

plt.imshow(
    abs(S_test),
    vmin=0,
    vmax=1,
    extent=[1, 2 * N, 2 * N, 1],
)
plt.colorbar()
plt.show()


xt = [i * 1000 for i in x]
yt = [i * 1000 for i in y]
plt.plot(xt, yt, "r .", label="diffuseur")
plt.xlim(0, 400)
plt.ylim(0, 100)
plt.legend()
plt.show()

t = Sf[N:,:N]
r = Sf[:N,:N]

t_moy = 0
r_moy = 0
for i in range(N):
    t_moy += t[i,i]
    r_moy += r[i,i]

def coef_np(M):
    M = abs(M)
    t = M[N:,:N]
    t = np.dot(t,np.conj(np.transpose(t)))
    r = M[:N,:N]
    r = np.dot(r,np.conj(np.transpose(r)))
    t_moy = 0
    r_moy = 0
    for i in range(N):
        t_moy += t[i,i]
        r_moy += r[i,i]
    return "T = "+str((t_moy/N))+ ", R = "+str((r_moy/N))

print(coef_np(Sf))
    
