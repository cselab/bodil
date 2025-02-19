import numpy as np
import torch

def upscale_grid(u, loc):
    dim = np.ndim(u)
    assert len(loc) == dim
    for l in loc:
        assert l in 'np'

    new_shape = []
    for s, l in zip(u.shape, loc):
        if l == 'n':
            new_shape.append(2 * s - 1)
        elif l == 'p':
            new_shape.append(2 * s)
    new_shape = tuple(new_shape)

    uf = torch.zeros(new_shape, dtype=u.dtype)

    uf[tuple([slice(None, None, 2)] * dim)] = u

    for axis, l in enumerate(loc):
        if l == 'n':
            slices_l = tuple(slice(None, -2, 2) if i == axis else slice(None) for i in range(dim))
            slices_r = tuple(slice(2, None, 2) if i == axis else slice(None) for i in range(dim))
            slices_mid = tuple(slice(1, -1, 2) if i == axis else slice(None) for i in range(dim))
            uf[slices_mid] = (uf[slices_l] + uf[slices_r]) / 2
        elif l == 'p':
            slices_src = tuple(slice(None, -1, 2) if i == axis else slice(None) for i in range(dim))
            slices_dst = tuple(slice(1, None, 2) if i == axis else slice(None) for i in range(dim))
            uf[slices_dst] = (uf[slices_src] + torch.roll(uf[slices_src], shifts=-1, dims=axis)) / 2
    return uf

class MultigridField:
    def __init__(self, u, loc, depth=1):
        self.depth = depth
        self.loc = loc
        dim = np.ndim(u)
        assert len(loc) == dim
        shape = np.array((np.shape(u)), dtype=int)
        self.mg = [u.clone()]
        for d in range(depth):
            for i, l in enumerate(l):
                s = shape[i]
                if l == 'n':
                    assert (s+1) % 2 == 0
                    assert s > 1
                    shape[i] = (s + 1) // 2
                elif l == 'p':
                    assert s % 2 == 0
                    assert s > 1
                    shape[i] = s // 2
            self.mg.append(torch.zeros(tuple(shape), dtype=u.dtype))

    def params(self):
        return self.mg

    def get(self):
        u = self.mg[-1]
        for subu in self.mg[-2::-1]:
            u = upscale_grid(u, loc=self.loc) + subu
        return u
