#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

def generate_samples_MCMC_(posterior, x0, num_samples, xmin, xmax, sigma=0.015, seed=239486):
    rng = np.random.default_rng(seed)
    x = x0.copy()
    p = posterior(x0)
    samples = []
    accepted = 0
    for k in range(num_samples):
        xp = rng.normal(x, sigma)
        xp = np.maximum(xmin, np.minimum(xmax, xp))
        pp = posterior(xp)
        u = rng.uniform()
        a = pp / p
        if u <= a:
            x = xp
            p = pp
            accepted += 1
        samples.append(x)
    arate = accepted / num_samples
    return arate, np.array(samples)

def generate_samples_MCMC(posterior, x0, num_samples, xmin, xmax, seed=239486, tolerance=0.05, max_iter = 100):
    target_rate = 0.65
    sigma_min = 0.0
    sigma_max = 1.0
    sigma_mid = (sigma_min + sigma_max) / 2
    arate_min, samples_min = generate_samples_MCMC_(posterior, x0, num_samples, xmin, xmax, sigma_min, seed)
    arate_max, samples_max = generate_samples_MCMC_(posterior, x0, num_samples, xmin, xmax, sigma_max, seed)
    arate_mid, samples_mid = generate_samples_MCMC_(posterior, x0, num_samples, xmin, xmax, sigma_mid, seed)

    print(f"sigma {sigma_mid} acceptance rate {arate_mid}")
    iter = 0
    while abs(target_rate - arate_mid) > tolerance and iter < max_iter:
        if arate_mid < target_rate:
            sigma_max = sigma_mid
            arate_max = arate_mid
            samples_max = samples_mid
        else:
            sigma_min = sigma_mid
            arate_min = arate_mid
            samples_min = samples_mid

        sigma_mid = (sigma_min + sigma_max) / 2
        arate_mid, samples_mid = generate_samples_MCMC_(posterior, x0, num_samples, xmin, xmax, sigma_mid, seed)
        iter += 1
        print(f"sigma {sigma_mid} acceptance rate {arate_mid}")

    return samples_mid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='csv files that contains losses against parameter x0')
    parser.add_argument('--sigma', type=float, default=0.1, help='data measurements error')
    parser.add_argument('--nsamples', type=int, default=1000, help='number of samples to generate')
    parser.add_argument('--out-csv', type=str, default="samples.csv", help='output path to dump generated samples')
    parser.add_argument('--show-plot', action='store_true', default=False, help='show plot')
    parser.add_argument('--show-samples', action='store_true', default=False, help='show samples on plot')
    args = parser.parse_args()

    nsamples = args.nsamples
    csv_path = args.csv
    n = 1024 # resolution for the interpolated function

    # ODIL loss has term lambda_data * mean(delta u^2), mean over nx * ny data points.
    # equivalent "likelihood" for data would be sum delta u^2 / 2 sigma^2, where sigma is measurements error
    # pseudo-likelihood is exp(-beta * ODLI_loss), so we can compute beta to correspond to a given sigma.
    nx = ny = 64
    lambda_data = 1
    sigma = args.sigma
    #beta = nx * ny / (2 * lambda_data * sigma**2)
    #beta = nx * ny / sigma
    beta = nx * ny / lambda_data / 10

    df = pd.read_csv(csv_path)
    x0 = df['x0'].to_numpy()
    y0 = df['y0'].to_numpy()
    ODIL_loss = df['ODIL_loss'].to_numpy()

    idx = np.lexsort((y0, x0))

    x0 = x0[idx]
    y0 = y0[idx]
    ODIL_loss = ODIL_loss[idx]

    interp = RBFInterpolator(np.vstack((x0, y0)).T,
                             ODIL_loss)


    xone = np.linspace(x0.min(), x0.max(), n, endpoint=True)
    yone = np.linspace(y0.min(), y0.max(), n, endpoint=True)

    X, Y = np.meshgrid(xone, yone, indexing='ij')
    dx = xone[1] - xone[0]
    dy = yone[1] - yone[0]

    loss = interp(np.vstack((X.flatten(), Y.flatten())).T).reshape((n, n))

    # for numerical stability
    loss -= np.min(loss)

    p = np.exp(-beta * loss)

    print(beta)
    print(np.min(p), np.max(p))
    norm = np.sum(p[:-1,:-1] + p[1:,:-1] + p[1:,1:] + p[-1:,1:]) * dx * dy / 4
    print(norm)
    p /= norm


    def posterior(x):
        l = interp(x.reshape((-1,2)))
        p = np.exp(-beta * l) / norm
        return p

    imax = np.argmax(p)
    x0max = np.array([X.flatten()[imax], Y.flatten()[imax]])

    samples = generate_samples_MCMC(posterior, x0max, num_samples=nsamples,
                                    xmin=[np.min(x0), np.min(y0)], xmax=[np.max(x0), np.max(y0)])

    data = {
        'x0': samples[:,0],
        'y0': samples[:,1]
    }
    pd.DataFrame(data).to_csv(args.out_csv, index=False)

    if args.show_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.contourf(X, Y, p, levels=100)
        if args.show_samples:
            ax.plot(samples[:,0], samples[:,1], '+r')
        ax.set_xlabel(r'$x_0$')
        ax.set_ylabel(r'$y_0$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
