#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=ituna
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:full:1
#SBATCH --ntasks=4
#SBATCH --output=slurm_output/worker_%A_%a_%j.out
#SBATCH --signal=TERM@300

### Setup Environment

# Set default conda environment name if not already set
if [ -z "$CONDA_ENV_NAME" ]; then
    CONDA_ENV_NAME="ituna"
fi

# Set default conda base path if not already set
if [ -z "$CONDA_BASE_PATH" ]; then
    CONDA_BASE_PATH="$HOME/miniconda3"
fi


source $CONDA_BASE_PATH/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME
echo "activating conda env: $CONDA_ENV_NAME"

conda env list
which python
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch; print('torch:is_cuda_available', torch.cuda.is_available())"
python -c "import numpy; print('numpy', numpy.__version__)"

nvidia-smi

nvidia-smi -L
# Get GPU UUIDs and store them in an array
output=$(nvidia-smi -L)
# Convert UUIDs into an array
readarray -t gpu_uuids < <(echo "$output" | grep "GPU" | awk -F'UUID: ' '{gsub(/\).*/, "", $2); print $2}')
num_gpus=${#gpu_uuids[@]}

echo "Available GPUs: ${num_gpus}"
echo "UUIDs: ${gpu_uuids[*]}"

# Create a comma-separated string of UUIDs for passing to srun
gpu_uuid_string=$(IFS=,; echo "${gpu_uuids[*]}")

# Echo the command that will be run
echo "Will run command:"
echo "$@"


# Launch multiple tasks and set CUDA_VISIBLE_DEVICES using UUID
srun --output slurm_output/log_%A_%a_%t.out \
    bash -c '
        IFS="," read -ra UUIDS <<< "'"$gpu_uuid_string"'"
        gpu_index=$((SLURM_LOCALID % ${#UUIDS[@]}))
        export CUDA_VISIBLE_DEVICES="${UUIDS[$gpu_index]}"
        echo "Task $SLURM_LOCALID assigned to GPU with UUID ${UUIDS[$gpu_index]}"
        exec "$@"
    ' bash "$@"

echo "Done"
