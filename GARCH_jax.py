import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

seed = 42
key = jax.random.key(seed)

N = int(2e3)

params_example = jnp.array([0.1, 0.05, 0.9])

def params_to_kurtosis(params, Y0=0, sigma20=0.1):

	omega = params[0]
	alpha = params[1]
	beta = params[2]

	def step(carry, z):
		Y, sigma2 = carry
		sigma2_next = omega + alpha * Y**2 + beta * sigma2
		Y_next = z * jnp.sqrt(sigma2_next)
		return (Y_next, sigma2_next), Y_next

	zs = jax.random.normal(key, shape=(N,))
	_, Y = jax.lax.scan(step, (Y0, sigma20), zs)

	X = jnp.arange(Y.shape[0])

	E_Y = jnp.mean(Y)
	Var_Y = jnp.mean((Y - E_Y)**2)
	Kurt_Y = jnp.mean((Y - E_Y)**4)

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

if __name__ == '__main__':
	params_to_kurtosis(params_example)