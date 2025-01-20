import re
import numpy as np

# 로그 파일 경로
log_file_path = "/home/hg-focal/isaacgym/IsaacGymEnvs/isaacgymenvs/logs/part_00138.log"

# Insertion Success 값을 저장할 리스트
insertion_success_values = []

# 로그 파일에서 Insertion Success 값 추출
with open(log_file_path, "r") as log_file:
    for line in log_file:
        match = re.search(r"Insertion Success:\s+([0-9\.]+)", line)
        if match:
            value = float(match.group(1))
            insertion_success_values.append(value)

# 평균과 표준편차 계산
if insertion_success_values:
    mean_success = np.mean(insertion_success_values)
    std_success = np.std(insertion_success_values)

    print(f"Insertion Success 평균: {mean_success:.4f}")
    print(f"Insertion Success 표준편차: {std_success:.4f}")
else:
    print("로그 파일에서 Insertion Success 값을 찾을 수 없습니다.")
