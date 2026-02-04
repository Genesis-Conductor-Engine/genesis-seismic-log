import unittest
import jax
import jax.numpy as jnp
from thrml_seismic_bridge import SeismicWrapper

class MockModel:
    def energy(self, state):
        return jnp.sum(state ** 2)

class MockSampler:
    def __init__(self):
        self.step_called = False

    def step(self, key, state):
        self.step_called = True
        return state * 0.5

class TestSeismicWrapper(unittest.TestCase):
    def test_run_protocol(self):
        model = MockModel()
        wrapper = SeismicWrapper(model)
        sampler = MockSampler()

        key = jax.random.PRNGKey(0)
        current_state = jnp.array([1.0, 1.0, 1.0])

        result = wrapper.run_protocol(key, sampler, current_state)

        self.assertTrue(sampler.step_called, "sampler.step should have been called")
        self.assertIn("status", result)
        self.assertIn("divergence", result)
        self.assertIn("energy_delta", result)

if __name__ == '__main__':
    unittest.main()
