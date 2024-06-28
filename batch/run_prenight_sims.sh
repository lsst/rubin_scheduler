#!/usr/bin/env bash
#SBATCH --account=rubin:developers      # Account name
#SBATCH --job-name=auxtel_prenight_daily   # Job name
#SBATCH --output=/sdf/data/rubin/user/neilsen/batch/auxtel_prenight_daily/daily.out # Output file (stdout)
#SBATCH --error=/sdf/data/rubin/user/neilsen/batch/auxtel_prenight_daily/daily.err  # Error file (stderr)
#SBATCH --partition=milano              # Partition (queue) names
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks run in parallel
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem=16G                       # Requested memory
#SBATCH --time=1:00:00                 # Wall time (hh:mm:ss)

echo "******** START of run_prenight_sims.sh **********"

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# SLAC S3DF - source all files under ~/.profile.d
if [[ -e ~/.profile.d && -n "$(ls -A ~/.profile.d/)" ]]; then
  source <(cat $(find -L  ~/.profile.d -name '*.conf'))
fi

__conda_setup="$('/sdf/group/rubin/user/neilsen/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sdf/group/rubin/user/neilsen/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/sdf/group/rubin/user/neilsen/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/sdf/group/rubin/user/neilsen/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/sdf/group/rubin/user/neilsen/mambaforge/etc/profile.d/mamba.sh" ]; then
    . "/sdf/group/rubin/user/neilsen/mambaforge/etc/profile.d/mamba.sh"
fi

mamba activate prenight
export AWS_PROFILE=prenight
WORK_DIR=$(date '+/sdf/data/rubin/user/neilsen/batch/auxtel_prenight_daily/%Y-%m-%dT%H%M%S' --utc)
echo "Working in $WORK_DIR"
mkdir ${WORK_DIR}
printenv > env.out
cd ${WORK_DIR}
prenight_sim --scheduler auxtel.pickle.xz --opsim None --repo "https://github.com/lsst-ts/ts_config_ocs.git" --script "Scheduler/feature_scheduler/auxtel/fbs_config_image_photocal_survey.py" --branch main
echo "******* END of run_prenight_sims.sh *********"

