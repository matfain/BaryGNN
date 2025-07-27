#!/bin/bash
#SBATCH --job-name=hello_world
#SBATCH --output=hello_world.out
#SBATCH --error=hello_world.err
#SBATCH --time=00:01:00
#SBATCH --partition=studentkillable 
#SBATCH --ntasks=1

python3 hello_world.py