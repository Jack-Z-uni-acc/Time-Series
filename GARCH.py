import numpy as np
import matplotlib.pyplot as plt

seed = 42

rng = np.random.default_rng(seed)

omega = 0.1
alpha = 0.5
beta = 0.4

N = int(1e3)

Y0 = 0
sigma20 = 0.1

Y = [Y0]
sigma2 = [sigma20]

for _ in range(N):

	z = rng.normal()
	sigma2_next = omega + alpha * Y[-1]**2 + beta * sigma2[-1]
	Y_next = z * np.sqrt(sigma2_next)
	Y.append(Y_next)
	sigma2.append(sigma2_next)

X = np.arange(len(Y))
Y = np.array(Y)

E_Y = np.mean(Y)
Var_Y = np.mean((Y - E_Y)**2)
Kurt_Y = np.mean((Y - E_Y)**4)

print('E[Y] = ', E_Y)
print('Var(Y) = ', Var_Y)
print('Kurt(Y) = ', Kurt_Y)

plt.plot(X, Y)
plt.show()