find . -name '*.mat' -print0 |
    xargs -0 -n 1 --process-slot-var I -P `nproc` sh -xc '
o=`basename $0 .mat`
taskset --cpu-list $I python -u main.py $0 --out-dir out/$o 2>out/$o/stderr 1>out/$o/stdout
echo $? > out/$o/status
'

