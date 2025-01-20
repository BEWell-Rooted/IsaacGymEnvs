import os

def run_command(command):
    """Helper function to run a shell command."""
    print(f"Executing: {command}")
    os.system(command)

def automate_training(part_numbers):
    """Automate the training process for multiple parts."""
    for part_number in part_numbers:
        print(f"Starting training for part: {part_number}")

        # Step 1: Collect disassembly path
        disassembly_command = (
            f"python train.py task=AutoMateTaskDisassemble "
            f"task.env.overwrite_subassemblies=True "
            f"task.env.desired_subassemblies=['asset_{part_number:05d}']"
        )
        run_command(disassembly_command)

        # Step 2: Start training
        training_command = (
            f"python train.py task=AutoMateTaskAssemble "
            f"task.env.overwrite_subassemblies=True "
            f"task.env.desired_subassemblies=['asset_{part_number:05d}'] "
            f"headless=True wandb_activate=True wandb_project='Automate Training'"
        )
        run_command(training_command)

        print(f"Completed training for part: {part_number}\n")

if __name__ == "__main__":
    # List of part numbers
    part_numbers = [
        138, 141, 143, 163, 175, 186, 187, 190, 192, 210, 
        211, 213, 255, 256, 271, 293, 296, 301, 308, 318
    ]

    # Start the automation
    automate_training(part_numbers)
