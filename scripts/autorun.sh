#!/bin/bash
TASK_NAME="$1"
MODE="$2"
EXTRA_ARG="$3"
echo "================================="
echo "Run task $TASK_NAME in mode $MODE"
echo "================================="

CONTAINER_NAME="genie_sim_benchmark"
START_SCRIPT="$PWD/scripts/start_gui.sh"
TERMINAL_ENV="autorun"
PROCESS_CLIENT="task_benchmark|teleop|replay|infer"
PROCESS_SERVER="isaac-sim|omni|omni_python|raise|ros"

SERVER_CONFIG=""


if [ "$TASK_NAME" = "clean" ]; then
    echo -e "\n\nEnter clean mode...\n\n"
    docker exec -it $CONTAINER_NAME bash -c "pkill -9 -f '$PROCESS_CLIENT\|$PROCESS_SERVER'" || true
    echo -e "Finish cleaning env..."
    docker exec -it $CONTAINER_NAME bash -c "find output -type f -name "*.db3" -exec rm -v {} \;" || true
    reset
    exit 0
fi

if [ "$TASK_NAME" = "genie_task_home_pour_water" ]; then
    SERVER_CONFIG="--enable_gpu_dynamics --render_mode RealTimePathTracing"
elif [ "$TASK_NAME" = "genie_task_home_wipe_dirt" ]; then
    SERVER_CONFIG="--enable_gpu_dynamics --render_mode RealTimePathTracing"
elif [ "$TASK_NAME" = "iros_pack_moving_objects_from_conveyor" ]; then
    SERVER_CONFIG="--reset_fallen=True"
fi

echo "Run with server config: $SERVER_CONFIG"

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    echo "Warning: Contrainer $CONTAINER_NAME not running, try to start..."

    if [ -x "$START_SCRIPT" ]; then
        echo "Executing script: $START_SCRIPT"
        "$START_SCRIPT"
        sleep 5
        if docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
            echo "Info: Container $CONTAINER_NAME started"
        else
            echo "Error: Container failed to start"
            exit 1
        fi
    else
        echo "Error:  Start script $START_SCRIPT not exist or not executable"
        exit 1
    fi
else
    echo "Info: Container $CONTAINER_NAME already running"
fi

if [ "$MODE" = "pico" ]; then
    echo -e "\n\nEnter pico teleop mode...\n\n"
    if [ -z "$3" ]; then
        echo "Error: IP address needed..."
        exit 1
    fi
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_teleop $TASK_NAME --mode pico --host_ip $EXTRA_ARG --record'"
    )
    declare -a DELAYS=(0 3)
elif [ "$MODE" = "keyboard" ]; then
    echo -e "\n\nEnter keyboard teleop mode...\n\n"
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_teleop $TASK_NAME --mode keyboard --record'"
    )
    declare -a DELAYS=(0 3)
elif [ "$MODE" = "infer" ]; then
    echo -e "\n\nEnter model infer mode...\n\n"
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_client $TASK_NAME'"
        "docker exec -it $CONTAINER_NAME bash -ic 'cd AgiBot-World && omni_python scripts/infer.py --task_name $TASK_NAME'"
    )
    declare -a DELAYS=(0 3 10)
elif [ "$MODE" = "replay" ]; then
    echo -e "\n\nEnter replay mode...\n\n"
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server --record_img --disable_physics --record_video $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_replay --task_file /root/workspace/main/benchmark/ader/eval_tasks/$TASK_NAME.json --state_file /root/workspace/main/$EXTRA_ARG --record'"
    )
    declare -a DELAYS=(0 3)
else
    echo -e "\n\nEnter empty benchmark mode...\n\n"
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_client $TASK_NAME'"
    )
    declare -a DELAYS=(0 3)
fi

TERMINAL_CMD=""
for term in gnome-terminal konsole xterm terminator; do
    if command -v "$term" &>/dev/null; then
        case "$term" in
        gnome-terminal) TERMINAL_CMD="gnome-terminal -- bash -c" ;;
        konsole) TERMINAL_CMD="konsole -e bash -c" ;;
        xterm) TERMINAL_CMD="xterm -e" ;;
        terminator) TERMINAL_CMD="terminator -e" ;;
        esac
        break
    fi
done

if [ -z "$TERMINAL_CMD" ]; then
    echo "No terminal emulator found. Please install one and try again."
    exit 1
fi

for i in "${!COMMANDS[@]}"; do
    sleep "${DELAYS[$i]}"
    if [[ "$TERMINAL_CMD" == "gnome-terminal"* ]]; then
        gnome-terminal -- bash -c "export TERMINAL_ENV=$TERMINAL_ENV; ${COMMANDS[$i]}; exec bash" &
    else
        $TERMINAL_CMD "${COMMANDS[$i]}" &
    fi
done

echo -e "\nAll terminal started, Press 'q' or 'Q' to stop all processes..."
while read -n 1 -s input; do
    if [[ "$input" == "q" || "$input" == "Q" ]]; then
        echo -e "\nSending Ctrl+C Signal..."
        docker exec -it $CONTAINER_NAME bash -c "pkill -SIGINT -f '$PROCESS_CLIENT'" || true
        break
    fi
done

reset
