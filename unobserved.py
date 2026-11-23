import numpy as np
import matplotlib.pyplot as plt

seed = 42
rng = np.random.default_rng(seed)

var_epsilon = 1
var_eta = 1

a1 = 0
P = 1

mu1 = rng.normal(a1, P)
y1 = 0

N = int(1e6)

mu = [mu1]
y = [y1]

delta_y = []

for _ in range(N):

	eta = rng.normal(0, var_eta)
	epsilon = rng.normal(0, var_epsilon)

	mu_next = mu[-1] + eta
	y_next = mu[-1] + epsilon

	delta_y_next = y_next - y[-1]

	mu.append(mu_next)
	y.append(y_next)

	delta_y.append(delta_y_next)

x = np.arange(len(y))
y = np.array(y)

E_delta_y = np.mean(np.array(delta_y))

print('Expectation of y increment: ', E_delta_y, '\n')

E_Y = np.mean(y)
Var_Y = np.mean((y - E_Y)**2)
Kurt_Y = np.mean((y - E_Y)**4)

print('E[Y] = ', E_Y)
print('Var(Y) = ', Var_Y)
print('Kurt(Y) = ', Kurt_Y)

plt.plot(x, y)
plt.show()