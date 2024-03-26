#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=MAF
#SBATCH --account rwth0934

#SBATCH --output=../output/MAF_%J.txt

#SBATCH --time=0-12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

source /home/zu992399/master_thesis/conda.sh
conda activate cathode


DIRVAR="/hpcwork/zu992399/bumphunt_paper/DE/development_blocks_2_transforms_5_hidden_256_lr_med/"
python MAF.py --input_set "baseline" --directory ${DIRVAR} --blocks 2 --transforms 5 --hidden 256 --learning_rate 1e-3 --batch_size 256

###cd ../../testing
###python run_pipeline.py --classifier "BDT" --mode "IAD" --directory "${DIRVAR}test/" --input_set "baseline"  --sample_test --samples_file "${DIRVAR}samples_inner.npy"
