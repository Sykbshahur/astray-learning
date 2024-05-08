import re
import os
import subprocess

# Function to run fairness_metrics.py


def run_fairness_metrics(output_txt):

    try:
        fairness_output = subprocess.run(
            ["python", "/home/ubuntu/shahur/Final_Misleading/new_metrics.py"],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running fairness_metrics.py: {e}")
        return

    # print("Debug: ", fairness_output.stdout)  # Debug print statement
    with open(output_txt, 'a') as f:
        f.write(f"Results for checkpoint: {checkpoint_file}\n")
        print(fairness_output.stdout)
        f.write(fairness_output.stdout)
        f.write("\n")


# Paths
checkpoint_folder = '/home/ubuntu/shahur/Final_Misleading/checkpoints/XceptionMis_Newthird_final_effic/ff++_XceptionMis_Newthird_final_M2/lamda1_0.1_lamda2_0.01_lr0.001/'
results_folder = '/home/ubuntu/shahur/Final_Misleading/checkpoints/XceptionMis_Newthird_final_effic/ff++_XceptionMis_Newthird_final_M2/result/out_res'
output_txt = '/home/ubuntu/shahur/Final_Misleading/checkpoints/XceptionMis_Newthird_final_effic/ff++_XceptionMis_Newthird_final_M2/result/result_celebdf.txt'

# List of attributes (replace with your actual attribute list)
interattributes = ['male,asian', 'male,black', 'male,others', 'male,white', 'nonmale,asian',
                   'nonmale,black', 'nonmale,white', 'nonmale,others']  # Replace with your actual list


# Get the list of files and filter out 'log_training.txt'
checkpoint_files = [f for f in os.listdir(
    checkpoint_folder) if f != 'log_training.txt']

# Sort the list numerically based on the numerical part of the string
# sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(
#     x[3:-4]) if x.startswith("ucf") and x.endswith(".pth") else 0)


def key_func(x):
    match = re.search(r'XceptionMis_Newthird_final_effic(\d+).pth', x)
    if match:
        return int(match.group(1))
    return 0


sorted_checkpoints = sorted(checkpoint_files, key=key_func)

print(sorted_checkpoints)
start_index = sorted_checkpoints.index('XceptionMis_Newthird_final_effic2.pth')
# Iterate through each checkpoint file
for checkpoint_file in sorted_checkpoints[start_index:]:
    # for checkpoint_file in sorted_checkpoints:
    if checkpoint_file == 'log_training.txt' or checkpoint_file == 'val_metrics.csv':
        continue  # Skip this file
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)

    if not os.path.isfile(checkpoint_path):
        continue

    # Run test.py
    subprocess.run(["python", "/home/ubuntu/shahur/Final_Misleading/training/test_2_mis.py", "--checkpoint", checkpoint_path])

    # Write checkpoint info to output_txt
    with open(output_txt, 'a') as f:
        f.write(f"Results for checkpoint: {checkpoint_file}\n")

    # Iterate through each attribute to find corresponding npy files and run fairness_metrics
    for eachatt in interattributes:
        labels_path = os.path.join(results_folder, f"{eachatt}labels.npy")
        predictions_path = os.path.join(
            results_folder, f"{eachatt}predictions.npy")

        # print(labels_path, predictions_path)

    # if os.path.exists(labels_path) and os.path.exists(predictions_path):
       
    else:
        with open(output_txt, 'a') as f:
            f.write(f"Missing .npy files for attribute: {eachatt}\n")
    run_fairness_metrics(output_txt)
print("Automated testing and fairness evaluation complete.")
