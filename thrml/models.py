import jax.numpy as jnp

class IsingEBM:
    """
    Mock implementation of Ising Energy Based Model.
    """
    def __init__(self, J=1.0, h=0.0):
        self.J = J
        self.h = h

    def energy(self, state):
        """
        Calculates the energy of the state.
        For a mock, we can just return sum of squares or something simple.
        """
        return jnp.sum(state ** 2)
