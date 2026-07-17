# B-ODIL

Code accompanying the paper:

> L. Amoudruz, S. Litvinov, C. Papadimitriou, P. Koumoutsakos,
> *Bayesian inference for PDE-based inverse problems using the optimization of a discrete loss*,
> Computer Methods in Applied Mechanics and Engineering **455** (2026) 118903.
> https://doi.org/10.1016/j.cma.2026.118903

It implements **B-ODIL**, a Bayesian extension of the Optimizing a DIscrete Loss
(ODIL) method for PDE-based inverse problems. B-ODIL treats the PDE loss as a
prior and combines it with a data likelihood to infer solutions with quantified
uncertainties, using either a Laplace approximation or Hamiltonian Monte Carlo
(HMC).

## Repository layout

- `bodil/` — core library (models, priors, likelihoods, samplers).
- `cases/` — self-contained scripts reproducing the paper's examples (see below).
- `poc/` — proof-of-concept and exploratory scripts.
- `test/` — tests.

## Cases

Each directory under `cases/` is a standalone example. Most contain the driver
scripts (`laplace.py`, `hmc.py`, `forward.py`, `inverse.py`, ...) and the outputs
used for the figures; several have their own `README.md` with specifics.

| Case | Description |
|------|-------------|
| `00_oscillator` | Harmonic oscillator (ODE) benchmark |
| `01_diffusion1D` | 1D diffusion equation |
| `02_hydrocube` | Flow reconstruction benchmark |
| `03_rbc` | Red blood cell shape inference |
| `04_reaction_diffusion` | Reaction–diffusion inverse problem |
| `05_glioma` | Glioma growth (voxel model) |
| `06_oscillator_omega` | Oscillator with unknown frequency |
| `07_reaction_diffusion_threshold` | Reaction–diffusion with level-set UQ |
| `08_pwip` | Pulse-wave imaging inverse problem |
| `09_gliodil` | 3D tumor concentration in a patient brain from MRI (GliODIL) |
| `10_oscillator_mesh_refinement` | Mesh-refinement study |
| `11_nonlinear_oscillator` | Nonlinear (Duffing) oscillator, prior-strength selection |

## Data

The datasets used by the cases live on the `data` branch and are fetched into
`data/` with:

```
make data
```

The glioma imaging data (MRI, FET-PET, and tissue segmentations) used by the
`05_glioma` and `09_gliodil` cases originates from the **GliODIL** dataset:
https://huggingface.co/datasets/m1balcerak/GliODIL. If you use it, please cite
the original work:

> M. Balcerak, J. Weidner, P. Karnakov, et al.,
> *Individualizing glioma radiotherapy planning by optimization of a data and
> physics-informed discrete loss*, Nature Communications **16**, 5982 (2025).
> https://doi.org/10.1038/s41467-025-60366-4

## Install

```
conda create -n bodil python=3.12
conda activate bodil
pip install torch dpdprops matplotlib pint pandas
conda install -c conda-forge mpi4py mpich
# optional:
pip install triangle
```

### FASRC (Harvard cluster)

- pyenv: https://docs.rc.fas.harvard.edu/kb/python-package-installation/
- pytorch: https://docs.rc.fas.harvard.edu/kb/pytorch/
- mpi4py: https://github.com/fasrc/User_Codes/tree/57c501d3a5925f81803f0c53ffba491be54c4c3b/Parallel_Computing/Python/mpi4py

```
module load gcc/14.2.0-fasrc01 openmpi/5.0.5-fasrc01 python/3.12.8-fasrc01
mamba create -n bodil python=3.12.8 numpy pip wheel
mamba install -n bodil -y cuda-toolkit=12.1.0 -c "nvidia/label/cuda-12.1.0"
mamba install -n bodil -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
mamba activate bodil
pip install dpdprops matplotlib pint pandas mpi4py nibabel
```

## Citation

```bibtex
@article{amoudruz2026bodil,
  title   = {Bayesian inference for PDE-based inverse problems using the optimization of a discrete loss},
  author  = {Amoudruz, Lucas and Litvinov, Sergey and Papadimitriou, Costas and Koumoutsakos, Petros},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {455},
  pages   = {118903},
  year    = {2026},
  doi     = {10.1016/j.cma.2026.118903}
}
```

## License

MIT — see [LICENSE](LICENSE).
