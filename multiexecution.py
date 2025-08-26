import subprocess
import re
import math
import matplotlib.pyplot as plt
from PIL import Image
import time

NUM_EXECUTIONS = 12

# Initialize lists to store the results of each execution
accuracies = []
recalls = []
false_positives = []
precisions = []
f1s = []
timeTraining = []
timeTest = []

# Path to save the combined images
output_combined_image = 'combined_plot.png'

# Run the script N times
for i in range(NUM_EXECUTIONS):
    print("Starting execution", i + 1)
    # Run the command and capture the output
    result = subprocess.run(['python3', 'hyperparameter_optimization.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Check if process has finished successfully
    if result.returncode != 0:
        print(f"Error in execution {i + 1}: {result.stderr}")
        continue
    output = result.stdout

    # Use regular expressions to extract metrics from the output
    accuracy_match = re.search(r'🟢 \| Accuracy:\s+([\d\.]+)', output)
    recall_match = re.search(r'🟢 \| Recall:\s+([\d\.]+)', output)
    false_positive_match = re.search(r'🟢 \| False Positive Rate:\s+([\d\.]+)', output)
    precision = re.search(r'🟢 \| Precision:\s+([\d\.]+)', output)
    training_time = re.search(r'OML Training time:\s+([\d\.]+)', output)
    test_time = re.search(r'Testing time:\s+([\d\.]+)', output)
    f1 = re.search(r'🟢 \| F1 Score:\s+([\d\.]+)', output)


    print("Execution", i + 1)
    print("Accuracy:", accuracy_match.group(1))
    print("Recall:", recall_match.group(1))
    print("False Positive Rate:", false_positive_match.group(1))
    print("Precision:", precision.group(1))
    print("F1 Score:", f1.group(1))
    print("Training time:", training_time.group(1))
    print("Test time:", test_time.group(1))

    # If the metrics are found, add them to the lists
    if accuracy_match and recall_match and false_positive_match:
        accuracy = float(accuracy_match.group(1))
        recall = float(recall_match.group(1))
        false_positive = float(false_positive_match.group(1))
        precision = float(precision.group(1))
        f1 = float(f1.group(1))
        training_time = float(training_time.group(1))
        test_time = float(test_time.group(1))

        accuracies.append(accuracy)
        recalls.append(recall)
        false_positives.append(false_positive)
        precisions.append(precision)
        f1s.append(f1)
        timeTraining.append(training_time)
        timeTest.append(test_time)

    # Copy and save the generated plot images
    image_filename = f'plot_acc_{i}.png'
    subprocess.run(['cp', 'plot_acc.png', image_filename])
    image_filename = f'plot_recall_{i}.png'
    subprocess.run(['cp', 'plot_recall.png', image_filename])
    image_filename = f'plot_fpr_{i}.png'
    subprocess.run(['cp', 'plot_fpr.png', image_filename])
    image_filename = f'plot_precision_{i}.png'
    subprocess.run(['cp', 'plot_precision.png', image_filename])
    image_filename = f'plot_f1_{i}.png'
    subprocess.run(['cp', 'plot_f1.png', image_filename])


print("Mean accuracy of", NUM_EXECUTIONS, "executions:", sum(accuracies) / len(accuracies))
print("Mean recall of", NUM_EXECUTIONS, "executions:", sum(recalls) / len(recalls))
print("Mean false positive rate of", NUM_EXECUTIONS, "executions:", sum(false_positives) / len(false_positives))
print("Mean precision of", NUM_EXECUTIONS, "executions:", sum(precisions) / len(precisions))
print("Mean f1 of", NUM_EXECUTIONS, "executions:", sum(f1s) / len(f1s))
print("Mean training time of", NUM_EXECUTIONS, "executions:", sum(timeTraining) / len(timeTraining))
print("Mean test time of", NUM_EXECUTIONS, "executions:", sum(timeTest) / len(timeTest))

# Now combine all 'plot_acc.png' images generated
images = [Image.open(f'plot_acc_{i}.png') for i in range(NUM_EXECUTIONS)]

# Assume all images have the same size
width, height = images[0].size

# Calculate number of rows and columns to combine images
num_columns = math.ceil(math.sqrt(NUM_EXECUTIONS))  # Square grid of executions
num_rows = math.ceil(NUM_EXECUTIONS / num_columns)

# Create a new image with the appropriate size to combine them
combined_image = Image.new('RGB', (width * num_columns, height * num_rows))

# Paste the images into the combined image
for i, image in enumerate(images):
    row = i // num_columns
    col = i % num_columns
    combined_image.paste(image, (col * width, row * height))

# Save the combined image
combined_image.save(output_combined_image)


# Save Recall plots
images = [Image.open(f'plot_recall_{i}.png') for i in range(NUM_EXECUTIONS)]
combined_image = Image.new('RGB', (width * num_columns, height * num_rows))
for i, image in enumerate(images):
    row = i // num_columns
    col = i % num_columns
    combined_image.paste(image, (col * width, row * height))
combined_image.save('combined_plot_recall.png')


# Save FPR plots
images = [Image.open(f'plot_fpr_{i}.png') for i in range(NUM_EXECUTIONS)]
combined_image = Image.new('RGB', (width * num_columns, height * num_rows))
for i, image in enumerate(images):
    row = i // num_columns
    col = i % num_columns
    combined_image.paste(image, (col * width, row * height))
combined_image.save('combined_plot_fpr.png')

# Save F1 plots
images = [Image.open(f'plot_f1_{i}.png') for i in range(NUM_EXECUTIONS)]
combined_image = Image.new('RGB', (width * num_columns, height * num_rows))
for i, image in enumerate(images):
    row = i // num_columns
    col = i % num_columns
    combined_image.paste(image, (col * width, row * height))
combined_image.save('combined_plot_f1.png')

# Save Precision plots
images = [Image.open(f'plot_precision_{i}.png') for i in range(NUM_EXECUTIONS)]
combined_image = Image.new('RGB', (width * num_columns, height * num_rows))
for i, image in enumerate(images):
    row = i // num_columns
    col = i % num_columns
    combined_image.paste(image, (col * width, row * height))
combined_image.save('combined_plot_precision.png')
