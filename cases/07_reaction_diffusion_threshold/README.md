# Reaction diffusion

Reaction diffusion equation with non uniform diffusion coefficient.
The goal is to reconstruct the concentration field given threshold concentration measurements, assuming we know the PDE and the form of the IC (but not the location of the IC).

## Forward

Generate data from forward problem:

```
./forward.py
```

## Inverse

Generate losses against varying parameters `x0`and `y0`

```
mpirun -n 8 ./run_plane.py
```

Generate posterior samples of (x_0, y_0):
```
./collect_losses.py out_plane/* --out-csv results/losses.csv
./generate_posterior_samples.py results/losses.csv --nsamples 128 --out-csv results/samples.csv
```

Predict concentrations from posteriors:
```
mpirun -n 8 ./run_samples.py results/samples.csv 
```



