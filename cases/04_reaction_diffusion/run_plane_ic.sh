#/bin/sh

dst=plane

mkdir -p $dst

for y0 in $(seq 0.050 0.05 0.950); do
    for x0 in $(seq 0.050 0.05 0.950); do
        echo $x0 $y0
        ./inverse.py \
            --out-dir $dst/x0_${x0}_y0_${y0} \
            --initial-pos $x0 $y0
    done
done
