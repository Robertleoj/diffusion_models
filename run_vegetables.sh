#!/bin/bash

#Basic
#SBATCH --account=gimli		# my group, Staff
#SBATCH --job-name=train_diffusion_animals   		# Job name
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb                     	# Job memory request
#SBATCH --partition=doTrain
#SBATCH --output=train_vegetables.log 	  	# Standard output and error log; +jobID

#TO USE    sbatch sbatchExample.sh

#date
RUNPATH=/home/gimli/diffusion_models/

cd $RUNPATH

source ./deepvenv/bin/activate
python3 train_vegetables.py
