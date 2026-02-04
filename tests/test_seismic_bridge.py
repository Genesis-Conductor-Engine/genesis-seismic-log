import jax
import jax.numpy as jnp
import unittest
import sys
import os

# Ensure we can import from root
sys.path.append(os.getcwd())

from thrml.models import IsingEBM
from thrml.sampling import GibbsSampler
from thrml_seismic_bridge import SeismicWrapper

class TestSeismicWrapper(unittest.TestCase):
    def test_run_protocol(self):
        # Setup
        model = IsingEBM()
        sampler = GibbsSampler()
        wrapper = SeismicWrapper(model)

        key = jax.random.PRNGKey(0)
        current_state = jnp.ones((10, 10))

        # Execution
        result = wrapper.run_protocol(key, sampler, current_state)

        # Verification
        print("Result:", result)
        self.assertIn("status", result)
        self.assertIn("divergence", result)
        self.assertIn("energy_delta", result)

        # Check shapes/types if possible
        # status should be a 0-d array (scalar)
        self.assertTrue(result["status"].shape == () or result["status"].shape == (1,))

if __name__ == '__main__':
    unittest.main()
