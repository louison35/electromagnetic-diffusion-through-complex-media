import torch
import torch.optim as optim
import numpy as np
import random as rd
import matplotlib.pyplot as plt

f = 7 * 10**9
omega = 2*np.pi*f
nx = 3
ny = 4
n = nx*ny
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


S0 = np.zeros((2 * N, 2 * N), dtype=complex)
G0 = [np.exp(1j * b * k[o]) for o in range(N)]
t0 = np.diag(G0)
S0[:N, N : 2 * N] = t0
S0[N : 2 * N, :N] = t0
S0 = torch.tensor(S0, dtype=torch.cfloat)

x = []
y = []

for i in range(nx):
    for j in range(ny):
        x.append(((i+1)*200/(nx+1))/1000)
        y.append(((j+1)*100/(ny+1))/1000)

x_diff = [rd.randrange(201,400) / 1000 for i in range(n_diff)]
y_diff = [rd.randrange(0, aa) / 1000 for i in range(n_diff)]

x_tot = x + x_diff
y_tot = y + y_diff

params = torch.tensor([6*torch.rand(1).item() for _ in range(len(x))], requires_grad=True)

def coef(M):
    M = abs(M)
    t = M[N:,:N]
    t = torch.matmul(torch.conj(torch.transpose(t,0,1)),t)
    r = M[:N,:N]
    r =torch.matmul(torch.conj(torch.transpose(r,0,1)),r)
    t_moy = 0
    r_moy = 0
    for i in range(N):
        t_moy += t[i,i]
        r_moy += r[i,i]
    return (t_moy/N),(r_moy/N)

par_diff = torch.tensor([-6 for i in range(n_diff)])

def matrix(par):
    par = torch.cat((par,par_diff))
    par1 = 1/-(par*1j)
    alpha = torch.diag(par1)
    S = S_matrix(x_tot, y_tot, alpha)-S0
    A= coef(S)[0]
    return 1/A

def matrice(par):
    par = torch.cat((par,par_diff))
    par1 = 1/-(par*1j)
    alpha = torch.diag(par1)
    return alpha

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
    return (t_moy/N),(r_moy/N)
    
optimizer = optim.SGD([params], lr=0.01)
num_iterations = 20000

for iteration in range(num_iterations):
    optimizer.zero_grad()
    
    cost = matrix(params) 
    cost.backward() 
    
    optimizer.step() 

    with torch.no_grad():
        params.clamp_(1e-6,100)

    if iteration % 100 == 0:
        print(f"Iteration {iteration}: <T> ={(1/cost).item():.4f}")

print(f"Final parameters: {params}")


S = (S_matrix(x_tot, y_tot, matrice(params))-S0).detach().numpy()
S_test = np.dot(np.conj(np.transpose(S)),S)
t = S[N:,:N]
t1 = np.dot(np.conj(np.transpose(t)),t)

plt.rcParams['figure.dpi'] = 300
plt.imshow(abs(t1),vmin = 0,vmax = 1)
plt.colorbar()
plt.show()



So = S_matrix(x_diff, y_diff, matrice(torch.tensor([])))-S0
print("T = ",coef_np(S)[0])
print("T0 = ",coef_np(So)[0])

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

print("x = ",x_tot)
print("y = ",y_tot)
print("params =",torch.cat((params,par_diff)).tolist())
