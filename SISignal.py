import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# create values of SI signal given by
# $ f(x) = \sum_{m\in\mathbb{Z}}d[m]a(t-mT)$
def si_signal(d: np.array, T: float, a, domain: np.arange):
    index = range(len(d))
    return np.array(
        [
            sum([d[m] * a(t - m * T) for m in index])
            for t in domain
        ]
    )

class SISignal:
    def __init__(self, d: np.array, T: float, a, domain: np.arange):
        self.x = domain
        self.y = si_signal(d, T, a, domain)

    def plot_signal(self):
        plt.plot(self.x, self.y)

# Example
domain = np.arange(-10, 10, .01)
signal_example = SISignal(
    [.1, 6, -.02, 1],
    0.2 * np.pi,
    np.sin,
    domain
)

print(f"values of {signal_example.x[::400]} are {signal_example.y[::400]}")
signal_example.plot_signal()
plt.savefig('../plot2.png')