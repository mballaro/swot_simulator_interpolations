#!/bin/bash

cd /home/ad/ballarm/

bash envvars.sh
. envvars.sh

python qsub_dask.py --nthreads 1 --nprocs 1 --memory-limit 16GB --interface ib0 --scale 50 --walltime 12:00:00 --profile envvars.sh

cd /home/ad/ballarm/tools/swotsimulator_interp

#/work/ALT/odatis/briolf/anaconda3/bin/python launcher.py setting_eNATL60-BLB002_science_sept2015_ell.py --first-date="2009-06-30T01:00:00.000000" --last-date="20100731" --scheduler-file="/home/ad/ballarm/scheduler.json" --debug


#/work/ALT/odatis/briolf/anaconda3/bin/python launcher.py setting_eNATL60-BLBT02_science_sept2015_ell.py --first-date="2009-06-30T01:00:00.000000" --last-date="20101029" --scheduler-file="/home/ad/ballarm/scheduler.json" --debug


/work/ALT/odatis/briolf/anaconda3/bin/python launcher.py setting_eNATL60-BLB002_calval_june2015_ell.py --first-date="2009-06-30T01:00:00.000000" --last-date="20100731" --scheduler-file="/home/ad/ballarm/scheduler.json" --debug


#/work/ALT/odatis/briolf/anaconda3/bin/python launcher.py setting_eNATL60-BLBT02_calval_june2015_ell.py --first-date="2009-06-30T01:00:00.000000" --last-date="20101029" --scheduler-file="/home/ad/ballarm/scheduler.json" --debug
