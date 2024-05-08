import re
import os
import subprocess
import sys
# Function to run fairness_metrics.py


# Paths
checkpoint_folder = '/home/ubuntu/shahur/Final_Misleading/checkpoints/XceptionMis_Newmethod_newscam_final/ff++_XceptionMis_Newmethod_Aug/lamda1_0.1_lamda2_0.01_lr0.001/'


# Get the list of files and filter out 'log_training.txt'
checkpoint_files = [f for f in os.listdir(
    checkpoint_folder) if f != 'log_training.txt']

# Sort the list numerically based on the numerical part of the string
# sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(
#     x[3:-4]) if x.startswith("ucf") and x.endswith(".pth") else 0)


def key_func(x):
    match = re.search(r'XceptionMis_Newmethod_newscam_final(\d+).pth', x)
    if match:
        return int(match.group(1))
    return 0


sorted_checkpoints = sorted(checkpoint_files, key=key_func)

print(sorted_checkpoints)
start_index = sorted_checkpoints.index('XceptionMis_Newmethod_newscam_final0.pth')
i = 0
# Iterate through each checkpoint file
for checkpoint_file in sorted_checkpoints[start_index:]:
    # for checkpoint_file in sorted_checkpoints:
    if checkpoint_file == 'log_training.txt' or checkpoint_file == 'val_metrics.csv':
        continue  # Skip this file
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)

    if not os.path.isfile(checkpoint_path):
        continue
    python_executable = sys.executable
    # Run test.py
    subprocess.run([python_executable, "/home/ubuntu/shahur/Final_Misleading/training/train_third_sam_final_auto.py", "--extractor_checkpoints", checkpoint_path, "--startepoch", str(i),"--epochs", str(i+1)])

    i += 1 
    # Write checkpoint info to output_txt
    
print("Automated testing and fairness evaluation complete.")
