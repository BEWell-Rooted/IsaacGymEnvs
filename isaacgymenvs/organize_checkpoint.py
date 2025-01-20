import os
import yaml
import json

# 학습 디렉토리 경로
runs_dir = "runs"
output_file = "checkpoints.json"

# 체크포인트 경로를 저장할 딕셔너리
checkpoints = {}

# 디렉토리 순회
for folder in os.listdir(runs_dir):
    folder_path = os.path.join(runs_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    # config.yaml 파일 확인
    config_path = os.path.join(folder_path, "config.yaml")
    if not os.path.exists(config_path):
        continue

    # config.yaml에서 desired_subassemblies 읽기
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    desired_subassemblies = config.get("task", {}).get("env", {}).get("desired_subassemblies", [])
    if not desired_subassemblies:
        continue

    # 부품 번호 추출
    part_number = desired_subassemblies[0].split("_")[-1]  # 'asset_00681' -> '00681'
    checkpoint_path = os.path.join(folder_path, "nn", "AutoMateTaskAssemble.pth")

    # 체크포인트 경로 확인
    if os.path.exists(checkpoint_path):
        checkpoints[part_number] = checkpoint_path

# 결과를 JSON 파일로 저장
with open(output_file, "w") as f:
    json.dump(checkpoints, f, indent=4)

print(f"Checkpoints saved to {output_file}")
