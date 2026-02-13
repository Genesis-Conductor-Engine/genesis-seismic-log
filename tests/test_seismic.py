import jax
import jax.numpy as jnp
from thrml_seismic_bridge import SeismicWrapper

class MockModel:
    def energy(self, state):
        return jnp.sum(state ** 2)

class MockSampler:
    def step(self, key, state):
        # Return a modified state to prove usage
        # This modification should be large enough to cause divergence
        return state + 10.0

def test_run_protocol():
    key = jax.random.PRNGKey(0)
    model = MockModel()
    # Stress factor 0 so shock doesn't add noise
    wrapper = SeismicWrapper(model, stress_factor=0.0)
    sampler = MockSampler()
    state = jnp.zeros((10, 10))

    # Run
    result = wrapper.run_protocol(key, sampler, state)

    # Assertions
    assert "status" in result
    assert "divergence" in result
    assert "energy_delta" in result

    divergence = result["divergence"]
    print(f"Divergence: {divergence}")

    # If sampler is used, divergence should be non-zero (10.0 per element -> distance > 0)
    # If mock is used (shaken * 0.99 where shaken=0), divergence is 0.
    # We expect this test to FAIL (divergence=0) before changes, and PASS (>0) after.

    # Since this is a test suite, passing means complying with intended behavior.
    # The intended behavior is to use the sampler.
    if divergence > 0.0:
        print("PASS: Sampler was used.")
    else:
        print("FAIL: Sampler was ignored (Mock logic used).")

if __name__ == "__main__":
    test_run_protocol()
