import torch
import matplotlib.pyplot as plt

n = 2
T = 1024

A = [2,1]
F = [1,3]
P = [0,0]
B = [0,0]

t = 2*torch.pi*torch.arange(T) / T

waves = []
for i in range(n):
    waves.append(A[i] * torch.sin(2*torch.pi*F[i]*t + P[i]) + B[i])

x = sum(waves) + 0*torch.rand((T,))

fig,ax = plt.subplots()
#ax.plot(x)
coefs = torch.fft.rfft(x)
power = coefs * torch.conj_physical(coefs) / T
ax.plot(power[1:])
plt.show()