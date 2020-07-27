import argparse
import getpass
import os
import pathlib
import psutil
import re
import subprocess
import sys
import tempfile
import dask.distributed
import dask.utils


def valid_time(s):
    match = re.search(r"(\d+):([0-5][0-9]):([0-5][0-9])", s)
    if match is None:
        raise argparse.ArgumentTypeError("Not a valid time: " + s)
    h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if h + m + s == 0:
        raise argparse.ArgumentTypeError(f"The duration cannot be zero: {s}")
    return f"{h:02d}:{m:02d}:{s:02d}"


def valid_name(s):
    error = len(s) > 15
    if not error:
        error = re.search(r'^[a-zA-Z][a-zA-Z0-9_]*$', s) is None
    if error:
        raise argparse.ArgumentTypeError("Invalid job name: " + s)
    return s


def valid_directory(s):
    if not pathlib.Path(s).is_dir():
        raise argparse.ArgumentTypeError(
            "The directory does not exist or is not readable: " + s)
    return s


def usage():
    parser = argparse.ArgumentParser(
        description="Submit a dask-driven python script to PBS.")
    parser.add_argument("--script", help="Python script to run")
    parser.add_argument("--nthreads",
                        metavar="INTEGER",
                        help="Number of threads per process.",
                        default=1,
                        type=int)
    parser.add_argument("--nprocs",
                        metavar="INTEGER",
                        help="Number of worker processes to launch.",
                        default=1,
                        type=int)
    parser.add_argument("--memory-limit",
                        metavar="INTEGER",
                        help="Bytes of memory per process that the worker "
                        "can use. This can be an integer (bytes) or string "
                        "(like 5GB or 5000M)",
                        default=None)
    parser.add_argument("--interface",
                        metavar="STRING",
                        help="Network interface like ‘eth0’ or ‘ib0’.",
                        default="eth0")
    parser.add_argument("--scale",
                        metavar="INTEGER",
                        help="Scale cluster to n workers",
                        default=1,
                        type=int)
    parser.add_argument("--walltime",
                        metavar="TIME",
                        help="max wallclock time",
                        type=valid_time,
                        default="01:00:00")
    parser.add_argument("--name",
                        metavar="IDENTIFIER",
                        help="job name",
                        type=valid_name,
                        default="dask")
    parser.add_argument(
        "--logdir",
        metavar="DIR",
        help="The directory containing the standard output and error logs",
        type=valid_directory,
        default=str(pathlib.Path().absolute()))
    parser.add_argument(
        "--env",
        metavar="CMD",
        help="Other commands to add to script before launching workers",
        nargs="+",
        default=None)
    parser.add_argument("--dry-run",
                        help="Display the job properties with no submission",
                        action="store_true")
    parser.add_argument("--profile",
                        help="Profile to use instead of .bash_profile",
                        default=pathlib.Path.home().joinpath(
                            pathlib.Path(".bash_profile")),
                        type=argparse.FileType(mode='r'))
    parser.add_argument("--kill", help="Kill PBS jobs", action="store_true")
    return parser.parse_args()


# int(TOTAL_MEMORY * min(1, ncores / _ncores))
def parse_memory_limit(memory_limit):
    if memory_limit is None:
        return psutil.virtual_memory().total()

    if isinstance(memory_limit, str):
        return dask.utils.parse_bytes(memory_limit)
    else:
        return int(memory_limit)


def kill(name):
    """Kill dask workers and cleanup the current directory"""
    username = getpass.getuser()
    output = subprocess.check_output('qstat', shell=True).decode('utf-8')
    for line in output.splitlines():
        if username in line and name in line:
            print(line)
            pid = line.split('.')[0]
            subprocess.call(f'qdel {pid}', shell=True)
    cwd = pathlib.Path.cwd()
    for item in os.listdir(cwd):
        if "dask-worker." in item:
            print(item)
            os.unlink(os.path.join(cwd, item))


def main():
    args = usage()

    if args.kill:
        kill(args.name)
        return

    memory = parse_memory_limit(args.memory_limit)
    memory_limit = int(memory / args.nprocs)

    source = ["source " + args.profile.name]
    if args.env is not None:
        source += args.env

    source = ";".join(source)
    prefix = "$ENV_SOURCE; "
    # prefix = "/home/ad/ballarm; "

    nprocs = max(args.nprocs, args.nthreads)

    if args.script is None:
        hms = [int(item) for item in args.walltime.split(":")]
        script = f"sleep {int(hms[0] * 3600 + hms[1] * 60 + hms[2]):d}"
    else:
        script = sys.executable + " " + args.script

    bash = f"""
#!/bin/bash
#PBS -N {args.name}
#PBS -l select={args.scale}:ncpus={nprocs}:mem={args.memory_limit}
#PBS -l walltime={args.walltime}
#PBS -e {args.logdir}/{args.name}_err.txt
#PBS -o {args.logdir}/{args.name}_out.txt

{source}
ENV_SOURCE="{source}"

SCHEDULER="$HOME/scheduler.json"
rm -f $SCHEDULER

export OMP_NUM_THREADS=1
NCPUS={args.nprocs} #Bug in NCPUS variable in our PBS install

INTERFACE="--interface {args.interface}"

NODES=$(cat $PBS_NODEFILE | wc -l)

# Run Dask Scheduler
echo "*** Launching Dask Scheduler ***"
pbsdsh -n 0 -- /bin/bash -c "{prefix}dask-scheduler $INTERFACE \\
    --scheduler-file $SCHEDULER  > /dev/null 2>&1;"&

# Run Dask workers
echo "*** Starting Workers on Other $NODES Nodes ***"
for ((ix=1; ix<=$NODES; ix+=1)); do
    pbsdsh -n ${{ix}} -- /bin/bash -c "{prefix}dask-worker $INTERFACE \\
        --scheduler-file $SCHEDULER \\
        --nthreads {args.nthreads} \\
        --nprocs {args.nprocs} \\
        --memory-limit {memory_limit} \\
        --local-directory $TMPDIR \\
        --name worker-{args.name}-${{ix}};"&
done

while [ ! -f $SCHEDULER ];
do
    sleep 1;
done;
sleep 1;

echo "*** Dask cluster is starting ***"
cd $PBS_O_WORKDIR/
{script}
"""
    for idx, item in enumerate(bash.split("\n")):
        print(f'  {idx:03d} {item}')

    if args.dry_run:
        return

    stream = tempfile.NamedTemporaryFile(mode="w")
    stream.write(bash)
    stream.flush()
    os.chmod(stream.name, 0o755)
    subprocess.check_call(["qsub", stream.name])


if __name__ == "__main__":
    main()


