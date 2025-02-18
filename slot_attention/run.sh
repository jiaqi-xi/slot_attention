#!/bin/bash

# SBATCH file can't directly take command args
# as a workaround, I first use a sh script to read in args
# and then create a new .slrm file for SBATCH execution

# read args from command line
GPUS=${GPUS:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}
PARTITION=$1
JOB_NAME=$2
PY_FILE=$3
LOG_DIR=$4

DATETIME=$(date "+%Y-%m-%d_%H:%M:%S")
LOG_FILE=$LOG_DIR/${DATETIME}.log

# set up log output folder
mkdir -p $LOG_DIR

# write to new file
echo "#!/bin/bash

# set up SBATCH args
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_DIR/slurm_output.log
#SBATCH --error=$LOG_DIR/slurm_error.log
#SBATCH --open-mode=append
#SBATCH --partition=$PARTITION                       # self-explanatory, set to your preference (e.g. gpu or cpu on MaRS, p100, t4, or cpu on Vaughan)
#SBATCH --cpus-per-task=$CPUS_PER_TASK               # self-explanatory, set to your preference
#SBATCH --ntasks-per-node=$GPUS
#SBATCH --mem=4G                                     # self-explanatory, set to your preference
#SBATCH --gres=gpu:$GPUS                             # NOTE: you need a GPU for CUDA support; self-explanatory, set to your preference 
#SBATCH --nodes=1
#SBATCH --qos=normal                                 # for 'high' and 'deadline' QoS, refer to https://support.vectorinstitute.ai/AboutVaughan2

# link /checkpoint to current folder
ln -sfn /checkpoint/\$USER/\$SLURM_JOB_ID \$PWD/checkpoint

# log some necessary environment params
echo \$SLURM_JOB_ID >> $LOG_FILE                      # log the job id
echo \$SLURM_JOB_PARTITION >> $LOG_FILE               # log the job partition

echo $CONDA_PREFIX >> $LOG_FILE                      # log the active conda environment 

python --version >> $LOG_FILE                        # log Python version
gcc --version >> $LOG_FILE                           # log GCC version
nvcc --version >> $LOG_FILE                          # log NVCC version

# run python file
python $PY_FILE $PY_ARGS >> $LOG_FILE                # the script above, with its standard output appended log file

" >> ./run.slrm


# run the created file
sbatch run.slrm

# delete it
# sleep 10
# rm -f run.slrm
