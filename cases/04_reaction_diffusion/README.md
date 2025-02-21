# Reaction diffusion

Reaction diffusion equation with non uniform diffusion coefficient.

## Forward

Generate and visualize forward problem:

```
./forward.py --dump-snapshots 
./make_movie.sh 
```

## Inverse

### Infer x0 only

Generate losses against varying parameter `x0`

```
run_line_ic.sh
./collect_losses.py out_inverse_* --out-csv results/losses_line.csv
```

Generate samples of `x0` according to posterior distribution
```
./generate_posterior_samples.py results/losses_line.csv --sigma 0.1 --nsamples 500
```


### Infer x0, y0

Generate losses against varying parameters `x0`and `y0`

```
run_plane_ic.sh
```
