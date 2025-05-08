# UQ ODIL

A proof-of-concept set of scripts for ODIL with uncertainty quantification.

# Data

```
make data
```

# Install

```
python -m pip install dpdprops matplotlib pint torch --user
```

# Environment

```
conda create -n uqodil python=3.12
pip install torch dpdprops matplotlib pint pandas
conda install -c conda-forge mpi4py mpich
```

## FASRC

* pyenv on fasrc: https://docs.rc.fas.harvard.edu/kb/python-package-installation/
* pytorch: https://docs.rc.fas.harvard.edu/kb/pytorch/
* mpi4py: https://github.com/fasrc/User_Codes/tree/57c501d3a5925f81803f0c53ffba491be54c4c3b/Parallel_Computing/Python/mpi4py

```
module load gcc/14.2.0-fasrc01 openmpi/5.0.5-fasrc01 python/3.12.8-fasrc01
mamba create -n uqodil python=3.12.8 numpy pip wheel
mamba install -n uqodil -y cuda-toolkit=12.1.0 -c "nvidia/label/cuda-12.1.0"
mamba install -n uqodil -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
mamba activate uqodil
pip install dpdprops matplotlib pint pandas mpi4py nibabel
```
