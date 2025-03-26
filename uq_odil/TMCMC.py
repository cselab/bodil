#!/usr/bin/env python

import numpy as np
from scipy.stats import norm, truncnorm
import scipy.optimize as optimize

def eval_log_like(samples, log_likelihood_func, comm, context):
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = len(samples)
    n_per_rank = (n + size - 1) // size
    start = n_per_rank * rank
    end = min([start + n_per_rank, n])
    rank_log_fvals = np.array([log_likelihood_func(samples[i], context) for i in range(start, end)])
    log_fvals = np.concatenate(comm.allgather(rank_log_fvals))
    return log_fvals


def TMCMC(log_likelihood,
          log_prior_density,
          prior_sampler,
          beta: float,
          gamma: float,
          num_samples: int,
          comm,
          seed):

    rank = comm.Get_rank()
    rng = np.random.default_rng(seed)

    zeta = 0
    S = 1.0
    stage = 1

    context = {'stage': stage}

    samples = np.array([prior_sampler(rng) for i in range(num_samples)])
    log_fvals = eval_log_like(samples, log_likelihood, comm, context)

    while zeta < 1:
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
        idx = rng.choice(np.arange(num_samples), size=num_samples, p=wj/np.sum(wj))

        # MCMC steps (we have hardcoded lmax=1 here)
        # see BASIS algorithm https://doi.org/10.1115/1.4037450

        samples = samples[idx]
        log_fvals = log_fvals[idx]
        log_f = np.array([zeta * f + log_prior_density(x) for x, f in zip(samples, log_fvals)])

        ## generate candidates
        candidate_samples = samples.copy()

        for k in range(num_samples):
            x = samples[k]
            xp = rng.multivariate_normal(mean=x, cov=cov)
            candidate_samples[k] = xp

        context = {'stage': stage}
        log_fvalsp = eval_log_like(candidate_samples, log_likelihood, comm, context)
        log_fp = np.array([zeta * fp + log_prior_density(xp) for xp, fp in zip(candidate_samples, log_fvalsp)])

        ## accept / reject

        accepted = log_fp >= log_f + np.log(rng.uniform(0, 1, size=num_samples))
        samples = np.where(accepted[:,None], candidate_samples, samples)
        log_fvals = np.where(accepted, log_fvals, log_fvalsp)

        if rank == 0:
            print(f"stage {stage}: zeta = {zeta}, S = {S}")

        stage += 1

    evidence = S
    return samples, evidence


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    def log_prior_density(x):
        return norm.logpdf(x[0], loc=0, scale=1) + norm.logpdf(x[1], loc=0, scale=1)

    def prior_sampler(rng):
        return rng.normal(size=2)

    def log_likelihood(x, context):
        return norm.logpdf(x[0], loc=1, scale=0.05) + norm.logpdf(x[1], loc=1, scale=0.2)

    samples, evidence = TMCMC(log_likelihood=log_likelihood,
                              log_prior_density=log_prior_density,
                              prior_sampler=prior_sampler,
                              beta=0.2,
                              gamma=1,
                              num_samples=128,
                              comm=comm,
                              seed=234987)

    if comm.Get_rank() == 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(*samples.T, "+k")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        plt.show()

if __name__ == '__main__':
    main()
