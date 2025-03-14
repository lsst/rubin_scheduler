#!/usr/bin/env bash
#SBATCH --account=rubin:developers      # Account name
#SBATCH --job-name=auxtel_prenight_daily   # Job name
#SBATCH --output=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims.out # Output file (stdout)
#SBATCH --error=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims.err  # Error file (stderr)
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

source /sdf/group/rubin/sw/w_latest/loadLSST.sh
conda activate /sdf/data/rubin/shared/scheduler/envs/prenight


TS_CONFIG_OCS_HASH=$(obs_version_at_time ts_config_ocs)
TS_FBS_UTILS_TAG=$(curl -s https://api.github.com/repos/lsst-ts/ts_fbs_utils/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)

export AWS_PROFILE=prenight
WORK_DIR=$(date '+/sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims/%Y-%m-%dT%H%M%S' --utc)
echo "Working in $WORK_DIR"
mkdir ${WORK_DIR}
cd ${WORK_DIR}

# Get ts_ocs_config
curl --output ts_ocs_config.zip https://github.com/lsst-ts/ts_config_ocs/archive/${TS_CONFIG_OCS_HASH}.zip
unzip ts_ocs_config.zip

# Install required python packages
PACKAGE_DIR=$(readlink -f ${WORK_DIR}/packages)
mkdir ${PACKAGE_DIR}

# Get the scheduler version from the EFD and install it.
RUBIN_SCHEDULER_TAG=v$(obs_version_at_time rubin_scheduler)
pip install --no-deps --target=${PACKAGE_DIR} git+https://github.com/lsst/rubin_scheduler.git@${RUBIN_SCHEDULER_TAG}

# Cannot get ts_fbs_utils from the EFD, so just guess the highest semantic version tag in the repo.
TS_FBS_UTILS_TAG=$(curl -s https://api.github.com/repos/lsst-ts/ts_fbs_utils/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
pip install --no-deps --target=${PACKAGE_DIR} git+https://github.com/lsst-ts/ts_fbs_utils.git@${TS_FBS_UTILS_TAG}

export PYTHONPATH=${PACKAGE_DIR}:${PYTHONPATH}

printenv > env.out
prenight_sim --scheduler auxtel.pickle.xz --opsim None --repo "https://github.com/lsst-ts/ts_config_ocs.git" --script "Scheduler/feature_scheduler/auxtel/fbs_config_image_photocal_survey.py" --branch main
echo "******* END of run_prenight_sims.sh *********"

