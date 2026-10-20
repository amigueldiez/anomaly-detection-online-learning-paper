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
VALIDATION_RATIO = config['VALIDATION_RATIO']
SELECTION_METRIC = config['SELECTION_METRIC']


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

#############################################################
# Dataset splitting and evaluation helpers                  #
#############################################################

# Metrics for which a lower value is better (used by select_best_parameter)
LOWER_IS_BETTER = {'false_positive', 'fpr'}


def split_flows(dataset, flows_train_scaler, flows_train_oml, validation_ratio=0.5, benign_additional=0):
    """Split a labeled flow dataset into scaler, train, validation and test sets.

    Benign flows are consumed in order to build the scaler and training (warm-up)
    sets. The remaining benign flows and all the anomalous flows are then divided
    between the validation and test sets. ``validation_ratio`` is the fraction of
    the anomalous flows assigned to the validation set; the rest go to the test
    set. Both sets are balanced so that they contain the same number of benign
    and anomalous flows (plus ``benign_additional`` extra benign flows). The four
    returned sets are pairwise disjoint.
    """
    if not 0.0 <= validation_ratio <= 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    benign = dataset.loc[dataset['Label'] == 0]
    anomalous = dataset.loc[dataset['Label'] == 1]

    n_benign_required = flows_train_scaler + flows_train_oml
    if len(benign) < n_benign_required:
        raise ValueError("Not enough benign flows to build the scaler and training sets")

    n_val_anomalies = int(round(len(anomalous) * validation_ratio))
    n_val_benign = n_val_anomalies + benign_additional
    n_test_benign = (len(anomalous) - n_val_anomalies) + benign_additional

    remaining_benign = benign.iloc[n_benign_required:]
    if len(remaining_benign) < n_val_benign + n_test_benign:
        raise ValueError("Not enough benign flows to balance the validation and test sets")

    validation = pd.concat([
        remaining_benign.iloc[:n_val_benign],
        anomalous.iloc[:n_val_anomalies],
    ])
    test = pd.concat([
        remaining_benign.iloc[n_val_benign:n_val_benign + n_test_benign],
        anomalous.iloc[n_val_anomalies:],
    ])

    return {
        'scaler': benign.iloc[:flows_train_scaler].sample(frac=1),
        'train': benign.iloc[flows_train_scaler:n_benign_required].sample(frac=1),
        'validation': validation.sample(frac=1),
        'test': test.sample(frac=1),
    }


def classification_metrics(labels, predictions):
    """Compute confusion-matrix counts and derived metrics.

    ``labels`` holds the ground-truth labels (1 = anomalous, 0 = benign) and
    ``predictions`` holds booleans where True means the flow was flagged as
    anomalous.
    """
    tp = tn = fp = fn = 0
    for label, predicted in zip(labels, predictions):
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 1:
            fn += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': (tp + tn) / total if total > 0 else 0,
        'precision': precision,
        'recall': recall,
        'false_positive': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
    }


def select_best_parameter(results, metric='f1'):
    """Return the parameter with the best ``metric`` on the validation set.

    ``results`` is a list of ``(parameter, metrics)`` pairs. For metrics such as
    the false positive rate, lower is better.
    """
    if not results:
        raise ValueError("No results to select from")
    if metric in LOWER_IS_BETTER:
        return min(results, key=lambda result: result[1][metric])[0]
    return max(results, key=lambda result: result[1][metric])[0]


# Dataset generation
def generate_dataset(filename, flows_train_scaler=FLOWS_TRAIN_SCALER,
                     flows_train_oml=FLOWS_TRAIN_OML,
                     validation_ratio=VALIDATION_RATIO,
                     benign_additional=BENIGN_FLOWS_ADDITIONAL,
                     save_logs=True):

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

    splits = split_flows(
        dataset,
        flows_train_scaler=flows_train_scaler,
        flows_train_oml=flows_train_oml,
        validation_ratio=validation_ratio,
        benign_additional=benign_additional,
    )
    dataset_train_scaler = splits['scaler'].reset_index(drop=True)
    dataset_train_oml = splits['train'].reset_index(drop=True)
    dataset_validation_labeled = splits['validation'].reset_index(drop=True)
    dataset_test_labeled = splits['test'].reset_index(drop=True)

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

        for i in range(len(dataset_validation_labeled)):
            first_two_elements = {j: str(dataset_validation_labeled.iloc[i, j]) for j in range(2)}
            encoder.learn_one(first_two_elements)
            first_two_elements = encoder.transform_one(first_two_elements)
            dataset_validation_labeled.iloc[i, 0] = first_two_elements[0]
            dataset_validation_labeled.iloc[i, 1] = first_two_elements[1]

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
    print('Number of anomalies in the validation dataset:\t', dataset_validation_labeled.loc[dataset_validation_labeled['Label'] == 1].shape[0])
    print('Number of normal samples in the validation dataset:\t', dataset_validation_labeled.loc[dataset_validation_labeled['Label'] == 0].shape[0])
    print('Number of anomalies in the test dataset:\t', dataset_test_labeled.loc[dataset_test_labeled['Label'] == 1].shape[0])
    print('Number of normal samples in the test dataset:\t', dataset_test_labeled.loc[dataset_test_labeled['Label'] == 0].shape[0])
    print('=====================================================================================================')

    if save_logs and 'accuracy' in LOGS_SAVE:
        save_data_dataset(filename_log[0]+"Accuracy.log", dataset_train_scaler, dataset_train_oml, dataset_validation_labeled, dataset_test_labeled)
        save_data_dataset(filename_log[1]+"Accuracy.log", dataset_train_scaler, dataset_train_oml, dataset_validation_labeled, dataset_test_labeled)
        save_data_dataset(filename_log[2]+"Accuracy.log", dataset_train_scaler, dataset_train_oml, dataset_validation_labeled, dataset_test_labeled)
    if save_logs and 'recall' in LOGS_SAVE:
        save_data_dataset("50_70Recall.log", dataset_train_scaler, dataset_train_oml, dataset_validation_labeled, dataset_test_labeled)
        save_data_dataset(filename_log[0]+"Recall.log", dataset_train_scaler, dataset_train_oml, dataset_validation_labeled, dataset_test_labeled)
        save_data_dataset(filename_log[1]+"Recall.log", dataset_train_scaler, dataset_train_oml, dataset_validation_labeled, dataset_test_labeled)
        save_data_dataset(filename_log[2]+"Recall.log", dataset_train_scaler, dataset_train_oml, dataset_validation_labeled, dataset_test_labeled)

    dataset_train_no_labels = dataset_train_oml.drop(columns=['Label']).to_numpy()
    dataset_validation_no_labels = dataset_validation_labeled.drop(columns=['Label']).to_numpy()
    dataset_test_no_labels = dataset_test_labeled.drop(columns=['Label']).to_numpy()
    dataset_train_scaler = dataset_train_scaler.drop(columns=['Label'])

    return {
        'scaler': dataset_train_scaler,
        'train': dataset_train_no_labels,
        'validation': dataset_validation_no_labels,
        'test': dataset_test_no_labels,
        'validation_labeled': dataset_validation_labeled,
        'test_labeled': dataset_test_labeled,
    }



def build_scaler(scaling_type):
    if scaling_type == 'MaxAbsScaler':
        return preprocessing.MaxAbsScaler()
    elif scaling_type == 'MinMaxScaler':
        return preprocessing.MinMaxScaler()
    elif scaling_type == 'Normalizer':
        return preprocessing.Normalizer()
    elif scaling_type == 'RobustScaler':
        return preprocessing.RobustScaler()
    elif scaling_type == 'StandardScaler':
        return preprocessing.StandardScaler()
    raise ValueError(f"Unknown scaling type: {scaling_type}")


# Scale datasets
def scale_datasets(parameter, splits):
    print("⏳ | Scaling Datasets")
    scaler = build_scaler(parameter.get_scaling_type())

    def as_dict(row):
        return {i: value for i, value in enumerate(row)}

    # The scaler is fitted on the scaler and training flows only, so that the
    # validation and test flows remain held out.
    for _, row in splits['scaler'].iterrows():
        scaler.learn_one(as_dict(row))

    scaler_dataset_train = []
    scaler_dataset_validation = []
    scaler_dataset_test = []

    # The training stream keeps updating the scaler (online scaling).
    for _, row in pd.DataFrame(splits['train']).iterrows():
        row = as_dict(row)
        scaler.learn_one(row)
        scaler_dataset_train.append(list(scaler.transform_one(row).values()))

    # Validation and test are only transformed, never used to fit the scaler.
    for name, target in (('validation', scaler_dataset_validation), ('test', scaler_dataset_test)):
        for _, row in pd.DataFrame(splits[name]).iterrows():
            target.append(list(scaler.transform_one(as_dict(row)).values()))

    print("✅ | Dataset Scaled")
    return {
        'train': scaler_dataset_train,
        'validation': scaler_dataset_validation,
        'test': scaler_dataset_test,
    }



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
    print("✅ | Training phase completed")
    return model, end_time - start_time


# Evaluate a model over a labeled stream
def evaluate_stream(model, dataset_no_labels, dataset_labeled):
    """Run the online evaluation loop over a labeled stream.

    Returns a ``(metrics, curves)`` pair. ``metrics`` holds the final
    confusion-matrix counts and the derived metrics, while ``curves`` holds the
    cumulative evolution of each metric across the stream.
    """
    labels = list(dataset_labeled['Label'])
    tp, tn, fp, fn = 0, 0, 0, 0
    predictions = []
    curves = {'accuracy': [], 'recall': [], 'false_positive': [], 'precision': [], 'f1': []}

    for i, row in enumerate(dataset_no_labels):
        if isinstance(row, list):
            row = {f'feature_{j}': value for j, value in enumerate(row)}

        score = model.score_one(row)
        anomalous = model.classify(score)
        predictions.append(anomalous)

        probability_preprocessing.learn_one({0: score})
        probability.update(probability_preprocessing.transform_one({0: score})[0])

        if not anomalous:
            model.learn_one(row)

        label = labels[i]

        if anomalous and label == 1:
            tp += 1
        elif not anomalous and label == 0:
            tn += 1
        elif anomalous and label == 0:
            fp += 1
        elif not anomalous and label == 1:
            fn += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        curves['accuracy'].append((tp + tn) / (tp + tn + fp + fn))
        curves['recall'].append(recall)
        curves['false_positive'].append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        curves['precision'].append(precision)
        curves['f1'].append(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)

    return classification_metrics(labels, predictions), curves


# Test model
def test_model(model, final_test_dataset, dataset_test_labeled, parameter):
    start_time = time.time()
    metrics, curves = evaluate_stream(model, final_test_dataset, dataset_test_labeled)
    end_time = time.time()
    print("Testing time: ", end_time - start_time)

    accuracy = metrics['accuracy']
    recall = metrics['recall']
    false_positive = metrics['false_positive']
    precision = metrics['precision']
    f1_score = metrics['f1']

    print("✅ | Testing phase completed")
    print("🟢 | Accuracy: ", accuracy)
    print("🟢 | Recall: ", recall)
    print("🟢 | False Positive Rate: ", false_positive)
    print("🟢 | Precision: ", precision)
    print("🟢 | F1 Score: ", f1_score)
    print("TP: ", metrics['tp'], "TN: ", metrics['tn'], "FP: ", metrics['fp'], "FN: ", metrics['fn'])

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

    save_plot_accuracies(filename_plot_acc, curves['accuracy'])
    save_plot_recall(filename_plot_recall, curves['recall'])
    save_plot_false_positive(filename_plot_fpr, curves['false_positive'])
    save_plot_f1_score("plot_f1.png", curves['f1'])
    save_plot_precision("plot_precision.png", curves['precision'])
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

def save_data_dataset(log_file, dataset_train_scaler, dataset_train_oml, dataset_validation, dataset_test):
    log = open(log_file, "w")
    log.write('✅ | Datasets ready' + " \n")

    log.write('Number of anomalies in the train Scaler dataset: '+ str(dataset_train_scaler.loc[dataset_train_scaler['Label'] == 1].shape[0])+ " \n" )
    log.write('Number of normal samples in the train Scaler dataset: '+ str(dataset_train_scaler.loc[dataset_train_scaler['Label'] == 0].shape[0])+ " \n" )
    log.write('Number of normal samples in the trainOML dataset: '+ str(dataset_train_oml.loc[dataset_train_oml['Label'] == 0].shape[0])+ " \n" )
    log.write('Number of anomaly samples in the trainOML dataset: '+ str(dataset_train_oml.loc[dataset_train_oml['Label'] == 1].shape[0])+ " \n" )
    log.write('Number of anomalies in the validation dataset: '+ str(dataset_validation.loc[dataset_validation['Label'] == 1].shape[0])+ " \n" )
    log.write('Number of normal samples in the validation dataset: '+ str(dataset_validation.loc[dataset_validation['Label'] == 0].shape[0])+ " \n" )
    log.write('Number of anomalies in the test dataset: '+ str(dataset_test.loc[dataset_test['Label'] == 1].shape[0])+ " \n" )
    log.write('Number of normal samples in the test dataset: ' + str(dataset_test.loc[dataset_test['Label'] == 0].shape[0])+ " \n" )
    log.write('=====================================================================================================' + " \n \n \n" )
    log.close()



def main():
    # Generate parameter objects
    array_parameters = define_parameters(FEATURES, scaling_type, nu_value, q_value, learning_rate, FLOWS_TRAIN_OML)
    # Generate datasets (scaler, train, validation and test are pairwise disjoint)
    splits = generate_dataset(FILENAME)

    time.sleep(3)
    print("Number of iterations: ", len(array_parameters))

    # Scaling only depends on the scaler type, so cache the scaled sets.
    scaled_cache = {}

    def scaled_splits(parameter):
        scaling_name = parameter.get_scaling_type()
        if scaling_name not in scaled_cache:
            scaled_cache[scaling_name] = scale_datasets(parameter, splits)
        return scaled_cache[scaling_name]

    # Phase 1: select the best hyperparameter configuration on the validation set.
    results = []
    for parameter in array_parameters:
        scaled = scaled_splits(parameter)
        model, _ = train_model_OML(scaled['train'], parameter)
        metrics, _ = evaluate_stream(model, scaled['validation'], splits['validation_labeled'])
        results.append((parameter, metrics))
        print("🔵 | Validation accuracy: ", metrics['accuracy'],
              "| Recall: ", metrics['recall'],
              "| False Positive Rate: ", metrics['false_positive'],
              "| Precision: ", metrics['precision'],
              "| F1 Score: ", metrics['f1'])

    best_parameter = select_best_parameter(results, SELECTION_METRIC)
    print("✅ | Best configuration selected on the validation set using", SELECTION_METRIC)
    print("    Scaling: ", best_parameter.get_scaling_type(),
          "| Q: ", best_parameter.get_q_value(),
          "| Nu: ", best_parameter.get_nu_value(),
          "| Learning_rate: ", best_parameter.get_learning_rate())

    # Phase 2: retrain the selected configuration and evaluate it on the test set.
    scaled = scaled_splits(best_parameter)
    model, training_time = train_model_OML(scaled['train'], best_parameter)
    print("OML Training time: ", training_time)
    test_model(model, scaled['test'], splits['test_labeled'], best_parameter)



if __name__ == '__main__':
    main()

