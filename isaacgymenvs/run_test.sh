#!/bin/bash

# 체크포인트 JSON 파일
CHECKPOINT_FILE="checkpoints.json"

# 로그 디렉토리 생성
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 전체 실행 로그 파일
MAIN_LOG_FILE="$LOG_DIR/main_test_log.txt"

# 기존 로그 파일 제거
if [ -f "$MAIN_LOG_FILE" ]; then
    rm "$MAIN_LOG_FILE"
fi

# 테스트 실행
python -c "
import json
import os

with open('$CHECKPOINT_FILE', 'r') as f:
    checkpoints = json.load(f)

for part_number, checkpoint in checkpoints.items():
    log_file = f'logs/part_{part_number}.log'
    cmd = f'python train.py task=AutoMateTaskAssemble task.env.overwrite_subassemblies=True task.env.desired_subassemblies=[\\\"asset_{part_number}\\\"] test=True task.env.if_eval=True task.env.numEnvs=100 headless=True checkpoint=\\\"{checkpoint}\\\" > \\\"{log_file}\\\" 2>&1'
    print(f'Executing: {cmd}')
    os.system(cmd)
    # 로그 파일 기록
    with open('$MAIN_LOG_FILE', 'a') as main_log:
        main_log.write(f'Completed: {cmd}\\n')
        main_log.write(f'Part log saved to: {log_file}\\n')
"
