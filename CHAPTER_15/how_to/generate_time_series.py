
import numpy as np 

def generate_time_series(batch_size, n_steps):
    freqs1, freqs2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin( (time - offsets1) * (freqs1 * 10 + 10 ))
    series += 0.5 * np.sin( (time - offsets2) * (freqs2 * 10 + 10 ))
    series += 0.1 * np.sin(np.random.rand(batch_size, n_steps) - 0.5)
    return series[... , np.newaxis].astype(np.float32)