#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm
import scipy.optimize as optimize

def TMCMC(log_likelihood,
          log_prior_density,
          prior_sampler,
          beta: float,
          gamma: float,
          num_samples: int,
          rng):
    zeta = 0
    S = 1.0
    j = 1

    samples = np.array([prior_sampler() for i in range(num_samples)])

    while zeta < 1:
        log_fvals = np.array([log_likelihood(theta) for theta in samples])

        # adapt zeta
        zeta0 = zeta

        def cv(z):
            a = np.exp((z-zeta0) * log_fvals)
            return np.std(a) / np.mean(a)

        res = optimize.fsolve(lambda z: cv(z[0]) - gamma, x0=[zeta])
        zeta = min([1.0, res[0]])

        # compute plausibility weights and update S
        wj = np.exp(log_fvals * (zeta-zeta0))
        S *= np.mean(wj)

        cov = beta**2 * np.cov(samples.T)

        # resample
        candidate_samples = samples.copy()
        candidate_log_fvals = log_fvals.copy()
        num_times_chosen = np.zeros(num_samples, dtype=int)

        for k in range(num_samples):
            i = rng.choice(np.arange(num_samples), p=wj/np.sum(wj))
            if num_times_chosen[i] > 0:
                # MH step
                x = candidate_samples[i]
                log_f = zeta * candidate_log_fvals[i] + log_prior_density(x)

                xp = rng.multivariate_normal(mean=x, cov=cov)
                log_fvalp = log_likelihood(xp)
                log_fp = log_prior_density(xp) + zeta *  log_fvalp

                if log_fp >= log_f + np.log(rng.uniform(0, 1)):
                    candidate_samples[i] = xp
                    candidate_log_fvals[i] = log_fvalp

            num_times_chosen[i] += 1

            samples[k] = candidate_samples[i].copy()
            log_fvals[k] = candidate_log_fvals[i]

        print(f"stage {j}: zeta = {zeta}, S = {S}")

        j += 1

    evidence = S
    return samples, evidence


def main():

    rng = np.random.default_rng(123456)

    def log_prior_density(x):
        return norm.logpdf(x[0], loc=0, scale=1) + norm.logpdf(x[1], loc=0, scale=1)

    def prior_sampler():
        return np.random.normal(size=2)

    def log_likelihood(x):
        return norm.logpdf(x[0], loc=1, scale=0.02) + norm.logpdf(x[1], loc=1, scale=0.2)

    samples, evidence = TMCMC(log_likelihood=log_likelihood,
                              log_prior_density=log_prior_density,
                              prior_sampler=prior_sampler,
                              beta=0.2,
                              gamma=1,
                              num_samples=512,
                              rng=rng)

    fig, ax = plt.subplots()
    ax.plot(*samples.T, "+k")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    main()
