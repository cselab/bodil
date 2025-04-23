# Gliodil

## data

https://huggingface.co/datasets/m1balcerak/GliODIL

```
wget "https://huggingface.co/datasets/m1balcerak/GliODIL/resolve/main/data_GliODIL_essential.zip?download=true" -O gliodil_data.zip
unzip -q gliodil_data.zip
```

## FASRC

```
mkdir -p $SCRATCH/uq-odil
wget "https://huggingface.co/datasets/m1balcerak/GliODIL/resolve/main/data_GliODIL_essential.zip?download=true" -O $SCRATCH/uq-odil/gliodil_data.zip
(cd $SCRATCH/uq-odil && unzip -q gliodil_data.zip)
rm -r $SCRATCH/uq-odil/__MACOSX
rm $SCRATCH/uq-odil/gliodil_data.zip
```
