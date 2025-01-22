#!/usr/bin/env python

import torch
from torch.optim.optimizer import Optimizer, required

class HMC(Optimizer):
    r"""Hamiltonian Monte-Carlo.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        dt (float): step size (required)
        M (float, optional): mass of the parameters (default: 1)
        seed (int, optional): random seed (default: 123456)
        device: torch device (default: cpu)
    """

    def __init__(self, params, dt=required, M=1.0, L=5, seed=123456, device='cpu'):
        defaults = dict()
        super(HMC, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "HMC doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self._device = device
        self._generator = torch.Generator(device=device).manual_seed(seed)
        self._dt = dt
        self._M = M
        self._L = L

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    def __setstate__(self, state):
        super(HMC, self).__setstate__(state)

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _gen_p(self):
        dev = self._device
        n = self._numel()
        M = self._M
        return torch.normal(mean=torch.zeros(n, device=dev),
                            std=torch.full((n,), M, device=dev),
                            generator=self._generator)


    def step(self, closure):
        """ Performs a single sampling step.
        Arguments:
            closure (callable): A closure that zeroes the gradient, evaluates U, compute gradient of U and returns U.
        """
        dev = self._device
        n = self._numel()
        M = self._M
        dt = self._dt

        p = self._gen_p()

        r_init = self._clone_param()
        U = closure()
        H0 = U + torch.sum(p**2) / (2 * M)
        gradU = self._gather_flat_grad()

        for i in range(self._L):
            with torch.no_grad():
                p -= dt/2 * gradU
                # r += dt/M * p
                self._add_grad(step_size=dt/M, update=p)

            U = closure()

            with torch.no_grad():
                gradU = self._gather_flat_grad()
                p -= dt/2 * gradU

        H = U + torch.sum(p**2) / (2 * M)
        u = torch.rand(size=(1,), generator=self._generator)[0].item()
        alpha = min([torch.exp(H - H0).item(), 1.0])

        if u <= alpha:
            # accept
            H_ = H.item()
            x_ = self._clone_param()
        else:
            # reject
            with torch.no_grad():
                self._set_param(r_init)
            H_ = H0.item()
            x_ = self._clone_param()

        return x_, H_
