#!/bin/bash
NUM="$1"
TASK_NAME="$2"
MODEL_NAME="$3"

printf "===================================\n"
printf " Evaluation\n"
printf " Task: $TASK_NAME\n Model: $MODEL_NAME\n Times: $NUM\n"
printf "===================================\n\n"

sleep 5
./scripts/autorun.sh clean

timeout=5
for i in $(seq 1 $NUM); do
    printf "\n\n  ROUND $i...\n\n"
    tmux new-session -s "term_$i" "
        ./scripts/autorun.sh '$TASK_NAME' infer '$MODEL_NAME'
    "
    end_time=$(( $(date +%s) + $timeout ))
    while (( $(date +%s) < end_time )); do
        left_time=$(( $end_time - $(date +%s) ))
        printf "\r Press q or Q to stop batch run in $left_time seconds...\n"
        read -t 1 -n 1 input
        if [[ "$input" == "q" || "$input" == "Q" ]]; then
            printf " Terminate batch run...\n"
            ./scripts/autorun.sh clean
            reset
            exit 1
        fi
    done
    printf " Continue batch run...\n"
    ./scripts/autorun.sh clean
done

printf "\n\nEvaluation finished. Cleaning env...\n\n"
reset
