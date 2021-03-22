#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --mail-user=dengyx93@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=%j.out


module load python/3.7.4
source $HOME/ENV/bin/activate
python $HOME/projects/def-jlevman/x2019/covid/covid.py
