import torch
import torch.optim as optim
import numpy as np
import random as rd
import matplotlib.pyplot as plt


f = 7 * 10**9
omega = 2*np.pi*f
n = 12
n_diff = 3
a = 0.1
aa = 100
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

# np.sqrt((omega / c0) ** 2 - ((i * np.pi) / a) ** 2)
# (2 * np.pi / c0) * np.sqrt(f**2 - (i * (c0 / (2 * a))) ** 2)
# ((i + 1) * np.pi) / a

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


def S_matrix(x1, y1, alpha, k1=k):
    G1 = torch.tensor(green_in(x1, y1, k1), dtype=torch.cfloat)
    G_t1 = torch.transpose(G1, 0, 1)
    G_dd1 = green_dd(x1, y1, k1)
    w1 = alpha - torch.tensor(G_dd1, dtype=torch.cfloat)
    w_inv1 = torch.linalg.inv(w1)
    S1 = torch.matmul(G1, torch.matmul(w_inv1, G_t1))
    return S1

def coef(M):
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
    return (t_moy/N),(r_moy/N)



S0 = np.zeros((2 * N, 2 * N), dtype=complex)
G0 = [np.exp(1j * b * k[o]) for o in range(N)]
t0 = np.diag(G0)
S0[:N, N : 2 * N] = t0
S0[N : 2 * N, :N] = t0
S0 = torch.tensor(S0, dtype=torch.cfloat)

x = [0.050,0.050,0.050,0.050,0.1,0.1,0.1,0.1,0.15,0.15,0.15,0.15]
y = [0.02,0.04,0.06,0.08,0.02,0.04,0.06,0.08,0.02,0.04,0.06,0.08]

x_diff = [rd.randrange(201,400) / 1000 for i in range(n_diff)]
y_diff = [rd.randrange(0, aa) / 1000 for i in range(n_diff)]


x_tot = x+x_diff
y_tot = y+y_diff

# Matrice identité
B = torch.eye(N, dtype=torch.cfloat)


params1 = [rd.random()*(6) for i in range(n)]



# Définition de la fonction qui génère la matrice A
def generate_matrix_A(params):
    params2 = [-6 for i in range(n_diff)]
    params3 = params+params2
    A = torch.zeros((len(params3), len(params3)), dtype=torch.cfloat)
    for i in range(len(params3)):
        A[i, i] = 1 / -(params3[i] * 1j)
    return A

alpha = generate_matrix_A(params1)



T = 0.00000001
Ti = 0
i = 0

for j in range(30000):
    i+= 1 
    print(i)
    params1 = [rd.random()*6 for i in range(n)]
    alpha = generate_matrix_A(params1)
    Sf = S_matrix(x_tot, y_tot, alpha)-S0
    T1 = coef(Sf)[0]
    if T1>Ti:
        parami = params1
        Ti = T1
    print(Ti)


    
    
alpha = generate_matrix_A(parami)
Si = S_matrix(x_tot, y_tot, alpha)-S0
alpha0 = generate_matrix_A([])
So = S_matrix(x_diff, y_diff, alpha0)-S0
ti = Si[N:,:N]
ti = torch.matmul(torch.conj(torch.transpose(ti,0,1)),ti)
S_test = np.dot(np.conj(np.transpose(Si)),Si)

    
for i in range(len(parami)):
    print(str(x_tot[i])+":"+str(y_tot[i])+"; alpha : "+str(params1[i]))
print("T = "+str(coef(Si)[0]))
print("T0 = "+str(coef(So)[0]))


plt.rcParams['figure.dpi'] = 300

plt.imshow(abs(ti), vmin=0, vmax=1)
plt.colorbar()
plt.show()

xt = [i * 1000 for i in x_tot]
yt = [i * 1000 for i in y_tot]

x_bleu = [i*1000 for i in x_diff]
y_bleu = [i*1000 for i in y_diff]
x_rouge = [i*1000 for i in x]
y_rouge = [i*1000 for i in y]
    


plt.plot(x_rouge, y_rouge, "r .", label="diffuseur optimisé")
plt.plot(x_bleu, y_bleu, "b .", label="diffuseur fixe")
plt.xlim(0, 400)
plt.ylim(0, 100)
plt.legend()
plt.show()


