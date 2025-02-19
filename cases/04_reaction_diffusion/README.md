# Reaction diffusion

Reaction diffusion equation with non uniform diffusion coefficient.

## Forward

Generate and visualize forward problem:

```
./forward.py --dump-snapshots 
./make_movie.sh 
```

## Inverse

Generate losses against varying parameter `x0`

```
run_line_ic.sh
./collect_losses.py out_inverse_* --out-csv results/losses.csv
```

Generate samples of `x0` according to posterior distribution
```
./generate_posterior_samples.py results/losses.csv --sigma 0.1 --nsamples 500
```
