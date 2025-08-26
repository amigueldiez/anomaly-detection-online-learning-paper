import river
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from river import preprocessing
from river import anomaly
from river import optim
from river import sketch
import matplotlib.pyplot as plt
import re
from parameters import Parameters
from itertools import product
import yaml
import ipaddress
import time


#############################################################
# Variables configuration                                   #
#############################################################

#############################################################

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


FILENAME = config['FILENAME']
filename_log = config['filename_log']
filename_plot_acc = config['filename_plot_accuracy']
filename_plot_recall = config['filename_plot_recall']
filename_plot_fpr = config['filename_plot_fpr']
filename_plot_anomalies = config['filename_plot_anomalies']
scaling_type = config['scaling_type']
nu_value = config['nu_value']
q_value = config['q_value']
learning_rate = config['learning_rate']
FEATURES = config['FEATURES']
FLOWS_TRAIN_SCALER = config['FLOWS_TRAIN_SCALER']
FLOWS_TRAIN_OML = config['FLOWS_TRAIN_OML']
TYPE_IP = config['TYPE_IP']
SIGNIFICANT_PORT = config['SIGNIFICANT_PORT']
LOGS_SAVE = config['logs_save']
BENIGN_FLOWS_ADDITIONAL = config['BENIGN_FLOWS_ADDITIONAL']


probability_preprocessing = preprocessing.MinMaxScaler()
probability = sketch.Histogram()


#############################################################
# Variables initialisation                                  #
#############################################################

def define_parameters(features_dataset, scaling_type, nu_value, q_value, learning_rate, n_flows_train_oml):
    combinations = product(scaling_type, nu_value, q_value, learning_rate)
    parameter_array = []
    for es, nu_val, q_val, l_rate in combinations:
        parameter = Parameters(features_dataset, es, FLOWS_TRAIN_SCALER, nu_val, q_val, l_rate, n_flows_train_oml, 0, 0, 0)
        parameter_array.append(parameter)
    return parameter_array

# Validate IPv4 address
def is_valid_ipv4(ip):
    ipv4_pattern = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
    return ipv4_pattern.match(ip) and all(0 <= int(part) <= 255 for part in ip.split('.'))

# Validate IPv6 address
def is_valid_ipv6(ip):
    ipv6_pattern = re.compile(r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$")
    return ipv6_pattern.match(ip) and all(0 <= int(part, 16) <= 65535 for part in ip.split(':'))

# Dataset generation
def generate_dataset(filename):

    print('⏳ | Loading dataset...')
    dataset = pd.read_csv(filename, index_col=False)
    print('✅ | Dataset loaded')

    print('⏳ | Dividing and preprocessing the dataset...')
    if 'Attack' in dataset.columns:
        dataset = dataset.drop(dataset.columns[[-1]], axis=1)

    dataset = dataset[FEATURES]
    dataset = dataset.iloc[:, :].values

    if TYPE_IP == 'IPv4' and 'IPV_SRC_ADDR' or 'IPv4_SRC_ADDR' in FEATURES and 'IPV_SRC_ADDR' or 'IPv4_SRC_ADDR' in FEATURES:
        dataset[:, 0] = list(map(ip_to_integer, dataset[:, 0]))
        dataset[:, 1] = list(map(ip_to_integer, dataset[:, 1]))
    elif TYPE_IP == 'IP_DOMAINS' and 'IPV_SRC_ADDR' or 'IPv4_SRC_ADDR' in FEATURES and 'IPV_SRC_ADDR' or 'IPv4_SRC_ADDR' in FEATURES:
        print("ℹ️ | MODE IP_DOMAINS ACTIVATED")
        for i in range(len(dataset)):
            if not re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', dataset[i][0]) and not re.match(r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))', dataset[i][0]):
                dataset[i][0] = dataset[i][0].split('.')[-2] + '.' + dataset[i][0].split('.')[-1]

        for i in range(len(dataset)):
            if not re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', dataset[i][1]) and not re.match(r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))', dataset[i][1]):
                dataset[i][1] = dataset[i][1].split('.')[-2] + '.' + dataset[i][1].split('.')[-1]

    if SIGNIFICANT_PORT:
        print("ℹ️ | MODE SIGNIFICANT PORT ACTIVATED")
        for i in range(len(dataset)):
            if dataset[i][FEATURES.index('DIRECTION')] == 1:
                dataset[i][FEATURES.index('DIRECTION')] = dataset[i][FEATURES.index('L_DST_PORT')]
            else:
                dataset[i][FEATURES.index('DIRECTION')] = dataset[i][FEATURES.index('L_SRC_PORT')]
        print('✅ | Significant port added')
        
    if "FIRST_SWITCHED" in FEATURES:
        for i in range(len(dataset)):
            dataset[i][FEATURES.index('FIRST_SWITCHED')] = int(dataset[i][FEATURES.index('FIRST_SWITCHED')].split(' ')[1].split(':')[0])

    dataset = pd.DataFrame(dataset, columns=FEATURES)

    if SIGNIFICANT_PORT:
        dataset = dataset.drop(columns=['L_SRC_PORT', 'L_DST_PORT'])
        dataset = dataset.rename(columns={'DIRECTION': 'PORT'})

    datasets_training = dataset.loc[dataset['Label'] == 0].iloc[:FLOWS_TRAIN_SCALER + FLOWS_TRAIN_OML]
    dataset_train_scaler = datasets_training.loc[dataset['Label'] == 0].iloc[:FLOWS_TRAIN_SCALER]
    dataset_train_oml = datasets_training.loc[dataset['Label'] == 0].iloc[FLOWS_TRAIN_SCALER:FLOWS_TRAIN_SCALER + FLOWS_TRAIN_OML + 1]

    dataset = dataset.drop(dataset_train_scaler.index)
    dataset = dataset.drop(dataset_train_oml.index)

    anomalies_samples = dataset.loc[dataset['Label'] == 1].shape[0]
    benign_samples = dataset.loc[dataset['Label'] == 0].shape[0]

    if benign_samples > anomalies_samples:
        dataset = dataset.drop(dataset.loc[dataset['Label'] == 0].index[:benign_samples - anomalies_samples - BENIGN_FLOWS_ADDITIONAL])
    else:
        dataset = dataset.drop(dataset.loc[dataset['Label'] == 1].index[:anomalies_samples - benign_samples])
        if BENIGN_FLOWS_ADDITIONAL > 0:
            print("Not enough benign samples to add")

    dataset_test_labeled = dataset

    dataset_train_scaler = dataset_train_scaler.sample(frac=1).reset_index(drop=True)
    dataset_train_oml = dataset_train_oml.sample(frac=1).reset_index(drop=True)
    dataset_test_labeled = dataset_test_labeled.sample(frac=1).reset_index(drop=True)

    if TYPE_IP == 'IP_DOMAINS' and 'IPV_SRC_ADDR' or 'IPv4_SRC_ADDR' in FEATURES and 'IPV_SRC_ADDR' or 'IPv4_SRC_ADDR' in FEATURES:
        encoder = preprocessing.OrdinalEncoder()

        for i in range(len(dataset_train_scaler)):
            first_two_elements = {j: str(dataset_train_scaler.iloc[i, j]) for j in range(2)}
            encoder.learn_one(first_two_elements)
            first_two_elements = encoder.transform_one(first_two_elements)
            dataset_train_scaler.iloc[i, 0] = first_two_elements[0]
            dataset_train_scaler.iloc[i, 1] = first_two_elements[1]

        for i in range(len(dataset_train_oml)):
            first_two_elements = {j: str(dataset_train_oml.iloc[i, j]) for j in range(2)}
            encoder.learn_one(first_two_elements)
            first_two_elements = encoder.transform_one(first_two_elements)
            dataset_train_oml.iloc[i, 0] = first_two_elements[0]
            dataset_train_oml.iloc[i, 1] = first_two_elements[1]

        for i in range(len(dataset_test_labeled)):
            first_two_elements = {j: str(dataset_test_labeled.iloc[i, j]) for j in range(2)}
            encoder.learn_one(first_two_elements)
            first_two_elements = encoder.transform_one(first_two_elements)
            dataset_test_labeled.iloc[i, 0] = first_two_elements[0]
            dataset_test_labeled.iloc[i, 1] = first_two_elements[1]

    print('✅ | Datasets ready')
    print('=====================================================================================================')
    print('Number of anomalies in the train scaler dataset:\t', dataset_train_scaler.loc[dataset_train_scaler['Label'] == 1].shape[0])
    print('Number of normal samples in the train scaler dataset:\t', dataset_train_scaler.loc[dataset_train_scaler['Label'] == 0].shape[0])
    print('Number of normal samples in the trainOML dataset:\t', dataset_train_oml.loc[dataset_train_oml['Label'] == 0].shape[0])
    print('Number of anomaly samples in the trainOML dataset:\t', dataset_train_oml.loc[dataset_train_oml['Label'] == 1].shape[0])
    print('Number of anomalies in the test dataset:\t', dataset_test_labeled.loc[dataset_test_labeled['Label'] == 1].shape[0])
    print('Number of normal samples in the test dataset:\t', dataset_test_labeled.loc[dataset_test_labeled['Label'] == 0].shape[0])
    print('=====================================================================================================')

    if 'accuracy' in LOGS_SAVE:
        save_data_dataset(filename_log[0]+"Accuracy.log", dataset_train_scaler, dataset_train_oml, dataset_test_labeled)
        save_data_dataset(filename_log[1]+"Accuracy.log", dataset_train_scaler, dataset_train_oml, dataset_test_labeled)
        save_data_dataset(filename_log[2]+"Accuracy.log", dataset_train_scaler, dataset_train_oml, dataset_test_labeled)
    if 'recall' in LOGS_SAVE:
        save_data_dataset("50_70Recall.log", dataset_train_scaler, dataset_train_oml, dataset_test_labeled)
        save_data_dataset(filename_log[0]+"Recall.log", dataset_train_scaler, dataset_train_oml, dataset_test_labeled)
        save_data_dataset(filename_log[1]+"Recall.log", dataset_train_scaler, dataset_train_oml, dataset_test_labeled)
        save_data_dataset(filename_log[2]+"Recall.log", dataset_train_scaler, dataset_train_oml, dataset_test_labeled)

    dataset_train_no_labels = np.delete(dataset_train_oml, -1, axis=1)
    dataset_test_no_labels = np.delete(dataset_test_labeled, -1, axis=1)
    dataset_train_scaler = dataset_train_scaler.drop(columns=['Label'])

    return dataset_train_scaler, dataset_train_no_labels, dataset_test_no_labels, dataset_test_labeled



# Scale datasets
def scale_datasets(parameter, dataset_train_scaler, dataset_train_no_labels, dataset_test_no_labels):
    print("⏳ | Scaling Datasets")
    if parameter.get_scaling_type() == 'MaxAbsScaler':
        scaler = preprocessing.MaxAbsScaler()
    elif parameter.get_scaling_type() == 'MinMaxScaler':
        scaler = preprocessing.MinMaxScaler()
    elif parameter.get_scaling_type() == 'Normalizer':
        scaler = preprocessing.Normalizer()
    elif parameter.get_scaling_type() == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
    elif parameter.get_scaling_type() == 'StandardScaler':
        scaler = preprocessing.StandardScaler()
    
    scaler_dataset_train = []
    scaler_dataset_test = []

    dataset_train_scaler = pd.DataFrame(dataset_train_scaler)
    dataset_train_no_labels = pd.DataFrame(dataset_train_no_labels)
    dataset_test_no_labels = pd.DataFrame(dataset_test_no_labels)

    for _, row in dataset_train_scaler.iterrows():
        row = {i: value for i, value in enumerate(row)}
        scaler.learn_one(row)

    for _, row in dataset_train_no_labels.iterrows():
        row = row.to_dict()
        scaler.learn_one(row)
        row = scaler.transform_one(row)
        scaler_dataset_train.append(list(row.values()))

    for _, row in dataset_test_no_labels.iterrows():
        row = row.to_dict()
        scaler.learn_one(row)
        row = scaler.transform_one(row)
        scaler_dataset_test.append(list(row.values()))

    print("✅ | Dataset Scaled")
    return scaler_dataset_train, scaler_dataset_test



# Train model
def train_model_OML(scaler_dataset_train, parameter):
    # Define the model
    model = anomaly.QuantileFilter(
            anomaly.OneClassSVM(nu=parameter.get_nu_value(), intercept_lr=optim.schedulers.InverseScaling(learning_rate=parameter.get_learning_rate())),
            q=parameter.get_q_value()
        )
    # Training phase of OML
    print("⏳ | Training OML")
    start_time = time.time()
    for row in scaler_dataset_train:
        # Convert row to dict with features
        row_dict = {f'feature_{i}': value for i, value in enumerate(row)}
        model.learn_one(row_dict)
        score = model.score_one(row_dict)
        probability_preprocessing.learn_one({0: score})
        probability.update(probability_preprocessing.transform_one({0: score})[0])
    end_time = time.time()
    print("OML Training time: ", end_time - start_time)
    print("✅ | Training phase completed")
    return model


# Test model
def test_model(model, final_test_dataset, dataset_test_labeled, parameter):
    fp, fn, tp, tn = 0, 0, 0, 0
    accuracies, recalls, false_positives, anomalies, precision_list, f1_scores = [], [], [], [], [], []

    start_time = time.time()

    for i, row in enumerate(final_test_dataset):
        if isinstance(row, list):
            row = {f'feature_{j}': value for j, value in enumerate(row)}

        score = model.score_one(row)
        anomalous = model.classify(score)
        anomalies.append(anomalous)

        probability_preprocessing.learn_one({0: score})
        probability.update(probability_preprocessing.transform_one({0: score})[0])
        rank = probability.cdf(probability_preprocessing.transform_one({0: score})[0])

        if not anomalous:
            model.learn_one(row)

        label = dataset_test_labeled.iloc[i, -1]

        if anomalous and label == 1:
            tp += 1
        elif not anomalous and label == 0:
            tn += 1
        elif anomalous and label == 0:
            fp += 1
        elif not anomalous and label == 1:
            fn += 1

        accuracies.append((tp + tn) / (tp + tn + fp + fn))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        false_positives.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        precision_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        f1_scores.append(2 * (precision_list[i] * recalls[i]) / (precision_list[i] + recalls[i]) if (precision_list[i] + recalls[i]) > 0 else 0)

    end_time = time.time()
    print("Testing time: ", end_time - start_time)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive = fp / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("✅ | Testing phase completed")
    print("🟢 | Accuracy: ", accuracy)
    print("🟢 | Recall: ", recall)
    print("🟢 | False Positive Rate: ", false_positive)
    print("🟢 | Precision: ", precision)
    print("🟢 | F1 Score: ", f1_score)
    print("TP: ", tp, "TN: ", tn, "FP: ", fp, "FN: ", fn)

    parameter.accuracy = accuracy
    parameter.recall = recall
    parameter.false_positive = false_positive

    if 'accuracy' in LOGS_SAVE:
        if 0.70 <= accuracy < 0.80:
            saveData(filename_log[0]+"Accuracy.log", parameter)
        elif 0.80 <= accuracy < 0.90:
            saveData(filename_log[1]+"Accuracy.log", parameter)
        elif accuracy >= 0.90:
            saveData(filename_log[2]+"Accuracy.log", parameter)
    if 'recall' in LOGS_SAVE:
        if 0.5 <= recall < 0.7:
            saveData("50_70Recall.log", parameter)
        if 0.70 <= recall < 0.80:
            saveData(filename_log[0]+"Recall.log", parameter)
        elif 0.80 <= recall < 0.90:
            saveData(filename_log[1]+"Recall.log", parameter)
        elif recall >= 0.90:
            saveData(filename_log[2]+"Recall.log", parameter)

    save_plot_accuracies(filename_plot_acc, accuracies)
    save_plot_recall(filename_plot_recall, recalls)
    save_plot_false_positive(filename_plot_fpr, false_positives)
    save_plot_f1_score("plot_f1.png", f1_scores)
    save_plot_precision("plot_precision.png", precision_list)
    # save_plot_anomalies(filename_plot_anomalies, anomalies)
    # save_plot_histogram()


def save_plot_accuracies(filename_plot,accuracies):
    plt.figure()
    plt.plot(accuracies)
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of the model')
    print("📊 | Saving plot")
    plt.savefig(filename_plot)

def save_plot_recall(filename_plot,recall):
    plt.figure()    
    plt.plot(recall)
    plt.xlabel('Samples')
    plt.ylabel('Recall')
    plt.title('Recall of the model')
    print("📊 | Saving plot")
    plt.savefig(filename_plot)

def save_plot_false_positive(filename_plot,false_positive):
    plt.figure()
    plt.plot(false_positive)
    plt.xlabel('Samples')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate of the model')
    print("📊 | Saving plot")
    plt.savefig(filename_plot)


def save_plot_f1_score(filename_plot,f1_score):
    plt.figure()
    plt.plot(f1_score)
    plt.xlabel('Samples')
    plt.ylabel('F1 Score')
    plt.title('F1 Score of the model')
    print("📊 | Saving plot")
    plt.savefig(filename_plot)

def save_plot_precision(filename_plot,precision):
    plt.figure()
    plt.plot(precision)
    plt.xlabel('Samples')
    plt.ylabel('Precision')
    plt.title('Precision of the model')
    print("📊 | Saving plot")
    plt.savefig(filename_plot)

def save_plot_anomalies(filename_plot,anomalies):
    # Plot only last 100 elements
    anomalies = anomalies[-100:]
    # Plot the anomalies that it is only true or false
    plt.figure()
    plt.plot(anomalies)
    plt.xlabel('Samples')
    plt.ylabel('Anomalies')
    plt.title('Anomalies of the model')
    print("📊 | Saving plot")
    plt.savefig(filename_plot)


def save_plot_histogram():
    # Extract bin ranges and frequencies
    bin_labels = [f"[{bin.left:.2f}, {bin.right:.2f}]" for bin in probability]
    frequencies = [bin.count for bin in probability]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(bin_labels, frequencies, color="skyblue", edgecolor="black")

    # Labels and title
    plt.xlabel("Score range", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Score Histogram", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig("histogram.png")
    plt.close()



# Function to transform an IP address to an integer
def ip_to_integer(ip):
    if not ip or not isinstance(ip, str):  # Verifica si es nulo o no es una cadena
        raise ValueError(f"Invalid IP: {ip}")

    try:
        ip_obj = ipaddress.ip_address(ip)
        return int(ip_obj)
    except ValueError:
        raise ValueError(f"Invalid IP format: {ip}")

    
# Save Data
def saveData(log_file, parameter):
    with open(log_file, "a") as log:
        # log.write("🟢 | " + str(i) + " samples processed | \n")
        log.write("Features: " + str(parameter.get_features_dataset()) + "\n")
        log.write("Scaling: " + parameter.get_scaling_type() + 
                  "; Train_Flows_Scaler:" + str(parameter.get_n_flows_train_scaler()) +
                  "; Train_Flows_OML:" + str(parameter.get_flows_train_oml()) +
                  "; Q:" + str(parameter.get_q_value()) +
                  "; Nu:" + str(parameter.get_nu_value()) +
                  "; Learning_rate:" + str(parameter.get_learning_rate()) + " \n")
        log.write("Accuracy: " + str(parameter.get_accuracy()) + "\n")
        log.write("Recall: " + str(parameter.get_recall()) + "\n")
        log.write("False Positive Rate: " + str(parameter.get_false_positive()) + "\n\n")
        log.write("########################################################################### \n\n")

def save_data_dataset(log_file,dataset_train_scaler,dataset_train_oml,dataset_test):
    log = open(log_file, "w")
    log.write('✅ | Datasets ready' + " \n")

    log.write('Number of anomalies in the train Scaler dataset: '+ str(dataset_train_scaler.loc[dataset_train_scaler['Label'] == 1].shape[0])+ " \n" )
    log.write('Number of normal samples in the train Scaler dataset: '+ str(dataset_train_scaler.loc[dataset_train_scaler['Label'] == 0].shape[0])+ " \n" )
    log.write('Number of normal samples in the trainOML dataset: '+ str(dataset_train_oml.loc[dataset_train_oml['Label'] == 0].shape[0])+ " \n" )
    log.write('Number of anomaly samples in the trainOML dataset: '+ str(dataset_train_oml.loc[dataset_train_oml['Label'] == 1].shape[0])+ " \n" )
    log.write('Number of anomalies in the test dataset: '+ str(dataset_test.loc[dataset_test['Label'] == 1].shape[0])+ " \n" )
    log.write('Number of normal samples in the test dataset: ' + str(dataset_test.loc[dataset_test['Label'] == 0].shape[0])+ " \n" )
    log.write('====================================================================================================='" \n \n \n" )



def main():
    old_scaler = ""
    # Generate parameter objects
    array_parameters = define_parameters(FEATURES, scaling_type, nu_value, q_value, learning_rate, FLOWS_TRAIN_OML)
    # Generate datasets
    dataset_train_scaler, dataset_train_OML, dataset_test, dataset_test_labeled = generate_dataset(FILENAME)

    time.sleep(3)
    print("Number of iterations: ", len(array_parameters))
    i = 0

    for parameter in array_parameters:
        if old_scaler != parameter.get_scaling_type():
            # Scale datasets
            scaler_dataset_train_OML, scaler_dataset_test = scale_datasets(parameter, dataset_train_scaler, dataset_train_OML, dataset_test)

        # Train the Online model
        model = train_model_OML(scaler_dataset_train_OML, parameter)
        test_model(model, scaler_dataset_test, dataset_test_labeled, parameter)
        # Update old scaler
        old_scaler = parameter.get_scaling_type()



main()

