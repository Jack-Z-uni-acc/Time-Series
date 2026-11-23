import numpy as np
import matplotlib.pyplot as plt

seed = 42
rng = np.random.default_rng(seed)

N = int(1e6)

theta = 0.5
sigma2 = 1

X0 = 0

k = 1

X = [X0]
epsilon = [rng.normal(0, sigma2)]

for _ in range(N):

	epsilon_next = rng.normal(0, sigma2)
	X_next = epsilon_next - theta * epsilon[-1]

	epsilon.append(epsilon_next)
	X.append(X_next)

t = np.arange(len(X))
X = np.array(X)

def gamma(k):
	return np.mean((X * np.roll(X, k))[k::] / (np.std(X)**2))

gammas = []

n = 100

for k in range(n):

	gammas.append(gamma(k))

gammas = np.array(gammas)
ks = np.arange(n)

plt.plot(ks, gammas)
plt.show()
plt.figure()

plt.plot(t, X)
plt.show()


