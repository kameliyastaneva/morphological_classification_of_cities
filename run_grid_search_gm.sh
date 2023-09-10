#!/bin/bash -l

# Batch script to run an OpenMP threaded job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=20:00:00

# Request 1 gigabyte of RAM for each core/trun_pca.shrun_pca.shhread
# (must be an integer followed by M, G, or T)
#$ -l mem=30G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=100G

# Set the name of the job.
#$ -N process_feature_engineering

# Request 50 cores.
#$ -pe smp 36

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID

#$ -wd /home/zcftkst/kaminka_dissertation_scripts_prod

module purge
module load default-modules
module remove compilers mpi
module load compilers/gnu/4.9.2
module load openssl/1.1.1u
module load python3/recommended


# 8. Run the application.
source env/bin/activate
python3 scripts/grid_search_gm.py