#!/bin/bash -l
#SBATCH -J rand_gan
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -A NAML0001
#SBATCH -p dav
#SBATCH -C casper
#SBATCH -o random_gan.log.%J
#SBATCH --gres=gpu:v100:4
#SBATCH --reservation anemone
export PATH="/bin:/usr/bin"
export HOME="/glade/u/home/dgagne/" 
source $HOME/.bash_profile
module purge
module load gnu/7.3.0 openmpi-x/3.1.0 python/3.6.4
ncar_pylib 20180801-DL
cd $HOME/deepsky/scripts
python -u gan_random_fields.py &> casper_gan.log 
