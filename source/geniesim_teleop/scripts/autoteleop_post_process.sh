#!/bin/bash
# Automatic data processing: run process_data.py inside the container to
# post-process recorded teleop data for the specified task.
# Usage: ./scripts/autoteleop_post_process.sh <task_name> [user_name]
#   task_name: required, corresponds to <repo>/output/recording_data/<task_name>
#   user_name: optional, default is GenieSim

set -e

# Container name + repo mount point — same convention as start_teleop.sh.
CONTAINER_NAME="${GENIESIM_CONTAINER:-geniesim3}"
REPO_IN_CONTAINER="${GENIESIM_WORKSPACE:-/workspace}"
TELEOP_APP="${REPO_IN_CONTAINER}/source/geniesim_teleop/src/geniesim_teleop/app"
# numpy 1.26 (apt) for cv2 / rosbags image decoding — see start_teleop.sh.
NUMPY1_PATH="/usr/lib/python3/dist-packages"
USER_NAME="${2:-GenieSim}"

usage() {
    echo "Usage: $0 <task_name> [user_name]"
    echo "  task_name  required, name of the recording data dir (<repo>/output/recording_data/<task_name>)"
    echo "  user_name  optional, default is GenieSim"
    exit 1
}

if [ -z "$1" ]; then
    echo "Error: task_name must be provided"
    usage
fi

TASK_NAME="$1"

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    echo "Error: container '$CONTAINER_NAME' is not running. Start it first (e.g. geniesim docker5.1 up),"
    echo "       or set GENIESIM_CONTAINER=<name>."
    exit 1
fi

echo "Processing data: task_name=$TASK_NAME, user_name=$USER_NAME"

# --- Ensure post-processing deps are present in the container --------------
# process_data pulls in extract_ros_bag (rosbags / rosbags-image) and
# sim_data_converter (h5py). These aren't baked into the image and aren't
# installed by start_teleop.sh (the live teleop loop doesn't need them), so a
# fresh container fails here with `No module named 'rosbags'`. Install them
# once, idempotently. They resolve against NumPy 2.x and don't pull open3d, so
# the global NumPy and the apt NumPy-1.26 PYTHONPATH shim both stay intact.
if ! docker exec "$CONTAINER_NAME" python3 -c 'import rosbags, rosbags.image, h5py' 2>/dev/null; then
    echo "Info: installing post-processing deps (rosbags, rosbags-image, h5py) into $CONTAINER_NAME ..."
    docker exec "$CONTAINER_NAME" python3 -m pip install rosbags rosbags-image h5py
fi

docker exec "$CONTAINER_NAME" bash -ic "
    source /opt/ros/jazzy/setup.bash
    source ${TELEOP_APP}/bin/env.sh
    export SIM_REPO_ROOT=${REPO_IN_CONTAINER}
    export PYTHONPATH=${NUMPY1_PATH}:\$PYTHONPATH
    PYTHONUNBUFFERED=1 python3 -u -m geniesim_teleop.data_recording.process_data --data_dir '$TASK_NAME' --user '$USER_NAME'
"
echo "Data processing completed."
