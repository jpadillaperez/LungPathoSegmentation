#!/bin/sh
#SBATCH --job-name=AUG-Patho  # Job name
#SBATCH --output=./output/logs/Pathology_Segmentation-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=./output/logs/Pathology_Segmentation-%A.err  # Standard error of the script
#SBATCH --time=2-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:RTX3090:1  # Number of GPUs (Don't use more than 1 unless you absolutely need it and know what you are doing)
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 12 per GPU)
#SBATCH --mem=30G  # Memory in GB (Don't use more than 48G per GPU unless you absolutely need it and know what you are doing)
# SBATCH --nodelist=ithor  # Node to run the job on (ithor or dagobah for 24Gb)

CURRENT_LOG_DIR=/home/guests/jorge_padilla/code/Augsburg/LungPathoSegmentation/output/logs
OLD_LOG_DIR=/home/guests/jorge_padilla/code/Augsburg/LungPathoSegmentation/output/logs/previous_logs

# Move old logs to the new directory except for the current log file
for file in $CURRENT_LOG_DIR/*.out $CURRENT_LOG_DIR/*.err; do
    if [[ $file != $CURRENT_LOG_DIR/Pathology_Segmentation-$SLURM_JOB_ID.out ]] && [[ $file != $CURRENT_LOG_DIR/Pathology_Segmentation-$SLURM_JOB_ID.err ]]; then
        mv "$file" "$OLD_LOG_DIR/"
    fi
done

# Set your wandb API key
export WANDB_API_KEY=$(cat ./wandb_api_key.txt)

# load python module
source /home/guests/jorge_padilla/miniconda3/etc/profile.d/conda.sh

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate FCDenseNet

# Run the command
python -u train.py