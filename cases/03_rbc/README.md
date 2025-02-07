# RBC

Run UQ
```
./uq_stretch.py --data-csv data/stretch/mills_reduced.csv 
```

Visualize distribution of shapes (sliced)
```
for i in $(seq 0 6); do ./plot_cross_sections.py samples_$i --out slice_$i.png; done
```
