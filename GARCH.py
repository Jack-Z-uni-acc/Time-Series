import numpy as np
import matplotlib.pyplot as plt

seed = 42

rng = np.random.default_rng(seed)

omega = 0.1
alpha = 0.05
beta = 0.9

N = int(2e3)

Y0 = 0
sigma20 = 0.1

Y = [Y0]
sigma2 = [sigma20]
zs = [0]

for _ in range(N):

	z = np.random.normal()
	sigma2_next = omega + alpha * Y[-1]**2 + beta * sigma2[-1]
	Y_next = z * np.sqrt(sigma2_next)
	Y.append(Y_next)
	sigma2.append(sigma2_next)
	zs.append(z)


X = np.arange(len(Y))
Y = np.array(Y)
sigma2 = np.array(sigma2)

E_Y = np.mean(Y)
Var_Y = np.mean((Y - E_Y)**2)
Kurt_Y = np.mean((Y - E_Y)**4)

print('E[Y] = ', E_Y)
print('Var(Y) = ', Var_Y)
print('Kurt(Y) = ', Kurt_Y)

plt.plot(X, Y)
plt.plot(X, np.sqrt(sigma2))
plt.show()
plt.figure()

Y2 = Y**2

def gamma(k):
	return np.mean((Y * np.roll(Y, k))[k::] / (np.std(Y)**2))

gammas = []

def gamma_z(k):
	return np.mean((zs * np.roll(zs, k))[k::] / (np.std(zs)**2))

gammas_z = []

n = 100

for k in range(n):

	gammas.append(gamma(k))
	gammas_z.append(gamma_z(k))

gammas = np.array(gammas)
gammas_z = np.array(gammas_z) * np.sqrt(omega / (1 - (alpha + beta))) + E_Y
ks = np.arange(n)

plt.plot(ks, gammas)
plt.plot(ks, gammas_z)
plt.show()