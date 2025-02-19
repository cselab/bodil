#/bin/sh

: ${y0=0.33333}

for x0 in $(seq 0.050 0.025 0.950); do
    echo $x0 $y0
    ./inverse.py \
         --out-dir out_inverse_x0_${x0}_y0_${y0} \
         --initial-pos $x0 $y0
done
