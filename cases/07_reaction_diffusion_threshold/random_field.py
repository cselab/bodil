#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def generate_random_field(nx, ny, smoothness, rng):
    noise = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    kx = np.fft.fftfreq(nx)[:, None]
    ky = np.fft.fftfreq(ny)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    filter_function = np.exp(- (k * smoothness) ** 2)
    filtered_noise = noise * filter_function
    field = np.fft.ifft2(filtered_noise).real
    return field

def main():
    rng = np.random.default_rng(seed=123456)
    field = generate_random_field(64, 64, smoothness=16, rng=rng)
    field = np.where(field > 0, 1, 0)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(field, origin='lower', extent=[0, 1, 0, 1])
    plt.show()

if __name__ == '__main__':
    main()
