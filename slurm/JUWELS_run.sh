#!/bin/bash
#SBATCH --account=hai_1101
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=ituna
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


# Create slurm_output directory if it doesn't exist
mkdir -p slurm_output

nvidia-smi

# Echo the command that will be run
echo "Will run command:"
echo "$@"

srun --output slurm_output/log_%A_%a_%t.out \
    "$@"
