# Gliodil

## data

https://huggingface.co/datasets/m1balcerak/GliODIL

```
wget "https://huggingface.co/datasets/m1balcerak/GliODIL/resolve/main/data_GliODIL_essential.zip?download=true" -O gliodil_data.zip
unzip -q gliodil_data.zip
```

## FASRC

```
mkdir -p $SCRATCH/bodil
wget "https://huggingface.co/datasets/m1balcerak/GliODIL/resolve/main/data_GliODIL_essential.zip?download=true" -O $SCRATCH/bodil/gliodil_data.zip
(cd $SCRATCH/bodil && unzip -q gliodil_data.zip)
rm -r $SCRATCH/bodil/__MACOSX
rm $SCRATCH/bodil/gliodil_data.zip
```
