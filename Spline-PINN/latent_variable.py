from numpy import np

class LatentVariable:

    def __init__(self, orders: list[int]):
        self.orders = orders

    def hidden_size(self) -> int:
        return np.prod([i+1 for i in self.orders])
    

