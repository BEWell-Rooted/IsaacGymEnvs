import os
import re
import numpy as np
import json

# 로그 파일이 저장된 디렉토리
log_dir = "logs"
# JSON 파일 저장 경로
output_json_path = "insertion_stats.json"

# 결과를 저장할 딕셔너리
results = {}

# 로그 파일 순회
for log_file_name in os.listdir(log_dir):
    if log_file_name.startswith("part_") and log_file_name.endswith(".log"):
        # 부품 번호 추출
        part_number = log_file_name.split("_")[1].split(".")[0]

        # 로그 파일 경로
        log_file_path = os.path.join(log_dir, log_file_name)

        # Insertion Success 값을 저장할 리스트
        insertion_success_values = []

        # 로그 파일에서 Insertion Success 값 추출
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                match = re.search(r"Insertion Success:\s+([0-9\.]+)", line)
                if match:
                    value = float(match.group(1)) * 100  # 백분율로 변환
                    insertion_success_values.append(value)

        # 평균과 표준편차 계산
        if insertion_success_values:
            mean_success = np.mean(insertion_success_values)
            std_success = np.std(insertion_success_values)

            # 결과 저장
            results[part_number] = {
                "mean": round(mean_success, 2),
                "std": round(std_success, 2)
            }
        else:
            results[part_number] = {
                "mean": None,
                "std": None
            }

# JSON 파일로 저장
with open(output_json_path, "w") as output_file:
    json.dump(results, output_file, indent=4)

print(f"Results saved to {output_json_path}")
