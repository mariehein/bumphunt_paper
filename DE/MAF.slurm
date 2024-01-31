#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=MAF
#SBATCH --account rwth0934

#SBATCH --output=../output/MAF_%J.txt

#SBATCH --time=0-12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

source /home/zu992399/master_thesis/conda.sh
conda activate cathode_torch

python MAF.py --input_set "baseline" --directory "results/baseline_default/"