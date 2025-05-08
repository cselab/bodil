find data/PWIP/patientData -name '*.mat' -print0 |
xargs -0 -n 1 --process-slot-var I -P 4 sh -xc '
o=`basename $0 .mat`
mkdir -p out/$o
cd out/$o
CUDA_VISIBLE_DEVICES=$I python -u ../../train.cse.py ../../$0 10000 10000 100 2>stderr 1>stdout
echo $? > status
'
