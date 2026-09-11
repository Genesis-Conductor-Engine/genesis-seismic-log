
class Hamiltonian:
    def __init__(self, biases, temperature):
        self.biases = biases
        self.temperature = temperature

class Device:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def sample(self, hamiltonian, num_samples=1000):
        # Mock sampling: return a list of random states based on biases
        # This is just a simulation for the mock
        import random
        results = []
        for _ in range(num_samples):
            # Create a mock state dict
            state = {}
            for key, bias in hamiltonian.biases.items():
                # bias -1.0 means preferred (lower energy), so higher probability of being 1 (if 1 is state)
                # This is a dummy logic, just returning random bits
                state[key] = random.choice([0, 1])
            results.append(state)
        return results
