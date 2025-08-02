#!/bin/bash
NUM="$1"
TASK_NAME="$2"
MODEL_NAME="$3"

if [ "$NUM" = "-1" ]; then
    MODE="Generalization"
    printf "===================================\n"
    printf " Evaluation\n"
    printf " Task: $TASK_NAME\n Model: $MODEL_NAME\n Mode: $MODE\n"
    printf "===================================\n\n"
else
    MODE="Repeat"
    printf "===================================\n"
    printf " Evaluation\n"
    printf " Task: $TASK_NAME\n Model: $MODEL_NAME\n Mode: $MODE\n Times: $NUM\n"
    printf "===================================\n\n"
fi

if [ "$TASK_NAME" = "all" ]; then
    key_pattern="iros"
else
    key_pattern="$TASK_NAME"
fi

split_name() {
  echo "$1" | sed -E '
    s#.*/##;
    s/(.*)_[^_]+$/\1/;
    t success;
    s/.*//;
    :success;
  '
}


sleep 5
./scripts/autorun.sh clean

directory="benchmark/ader/eval_tasks/task_gen/"
declare -a file_list
if [ "$MODE" = "Generalization" ]; then
    while read -r file; do
        real_file=$(realpath "$file")
        file_list+=("$real_file")
    done < <(find "$directory" -type f -exec grep -l "$key_pattern" {} \; 2>/dev/null)
    CASE_NUM=${#file_list[@]}
else
    CASE_NUM=$NUM
fi

printf "\nTotal task number: $CASE_NUM\n"

timeout=5
for i in $(seq 1 $CASE_NUM); do
    if [ "$NUM" = "-1" ]; then
        printf "\n\n  Evaluation on ${file_list[$i-1]}...\n"
        TASK_NAME=$(split_name "${file_list[$i-1]}")
        printf "\n Extract task name $TASK_NAME from ${file_list[$i-1]}\n"
        cp "${file_list[$i-1]}" "benchmark/ader/eval_tasks/$TASK_NAME.json"
    else
        printf "\n\n  ROUND $i...\n\n"
    fi
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
            # reset
            exit 1
        fi
    done
    printf " Continue batch run...\n"
    ./scripts/autorun.sh clean
done

printf "\n\nEvaluation finished. Cleaning env...\n\n"
# reset
