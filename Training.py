import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def file_walker(root: str):
    """
    function to return list of all files in a nested directory
    """
    files = []
    for ze_root, _, ze_files in os.walk(root):
        for afile in ze_files:
            files.append(os.path.join(ze_root, afile))
    return files


def custom_cm(cm):
    """
    Custom Confusion Matrix Analyse
    """
    recall    = []
    precision = []
    f1        = []
    for i in range(cm.shape[0]):

        recall.append(cm[i][i]/cm.sum(axis=0)[i])
        precision.append(cm[i][i]/cm.sum(axis=1)[i])
        f1.append( 2 * (
            ( (cm[i][i]/cm.sum(axis=0)[i]) * (cm[i][i]/cm.sum(axis=1)[i]) )
                /
            ( (cm[i][i]/cm.sum(axis=0)[i]) + (cm[i][i]/cm.sum(axis=1)[i]) )
        ))

    recall    = [x for x in recall if str(x) != 'nan']
    precision = [x for x in precision if str(x) != 'nan']
    f1        = [x for x in f1 if str(x) != 'nan']

    return [
        sum(recall)/len(recall),
        sum(precision)/len(precision),
        sum(f1)/len(f1)
    ]



def train_model(
        dataset_path, n_epochs, loader,
        model, optimizer, criterion,
        use_cuda, save_path, model_name,
        logger,
    ):

    ####################################
    # log information about the dataset,
    #   the train, test, split ratios
    ####################################

    logger.log("-"*60)
    logger.log("Dataset Train Valid Test Split:")
    files = file_walker(dataset_path)
    dataset_type_dict = {}
    for afile in files:
        if afile.startswith("."): bfile = afile.split("\\", 1)[-1]
        else: bfile = afile
        data_type = bfile.split("\\")[1]
        label = bfile.split("\\")[2]
        out_lab = f"{label:<15s} || {data_type:>5s}"
        if out_lab not in dataset_type_dict:
            dataset_type_dict[out_lab] = []
        dataset_type_dict[out_lab].append(afile)
    logger.log("-"*60)
    for out_lab in dataset_type_dict:
        logger.log(f"--> {out_lab:<20s} || {len(dataset_type_dict[out_lab]):>3}")
    logger.log("-"*60)
    logger.log('\n--------------------------------------------------------------\n')

    ####################################
    # initiate the paths and the
    #   variables to save the training
    #   metrics in
    ####################################

    score_save_path = f"{save_path}\\score"
    if not os.path.exists(score_save_path):
        os.makedirs(score_save_path)
    model_save_path = f"{save_path}\\models"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_accuracy_list = []
    train_loss_list = []

    train_recall_list = []
    train_precision_list = []
    train_f1score_list = []

    valid_accuracy_list = []
    valid_loss_list = []

    valid_recall_list = []
    valid_precision_list = []
    valid_f1score_list = []

    metric_max = {
        'valid_accuracy_max': 0.0,
        'valid_recall_max': 0.0,
        'valid_precision_max': 0.0,
        'valid_f1score_max': 0.0
    }

    metric_max_at_epoch = {
        'Validation Accuracy': 0,
        'Validation Recall': 0,
        'Validation Precision': 0,
        'Validation F1 Score': 0
    }

    train_Confusion_matrix = np.zeros((4, 4))
    valid_Confusion_matrix = np.zeros((4, 4))

    ####################################
    # start the actual training
    ####################################

    for epoch in range(1, (n_epochs+1)):

        # print(f"EPOCH --> {str(epoch).zfill(4)}")

        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        model.train()
        start = time.time()

        ####################################
        # feed the training set images
        #   as batches to the model
        ####################################

        for batch_idx, (data, target) in enumerate(loader['train']):
            target_1 = target
            if use_cuda:
                data, target_2 = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target_2)
            loss.backward()
            optimizer.step()
            train_Confusion_matrix += confusion_matrix(
                preds.cpu().detach().numpy(),
                target_1.numpy(),
                labels=[i for i in range(4)]
            )
            train_acc = train_acc + torch.sum(preds == target_2.data)
            train_loss = train_loss + (
                (1 / (batch_idx + 1)) * (loss.data - train_loss)
            )

        model.eval()

        ####################################
        # evaluate the trained model
        #   using the validation set to
        #   adjust the hyperparameters
        ####################################

        for batch_idx, (data, target) in enumerate(loader['valid']):

            target_1 = target
            if use_cuda:
                data, target_2 = data.cuda(), target.cuda()
            output = model(data)

            _, preds = torch.max(output, 1)
            loss = criterion(output, target_2)

            valid_Confusion_matrix += confusion_matrix(
                preds.cpu().detach().numpy(),
                target_1,
                labels=[i for i in range(4)]
            )

            valid_acc = valid_acc + torch.sum(preds == target_2.data)
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss.data - valid_loss)
            )

        ####################################
        # calculate all the evaluation
        #   metrics for the model on both
        #   the validation set and the
        #   training set
        ####################################

        train_loss = train_loss/len(loader['train'].dataset)
        valid_loss = valid_loss/len(loader['valid'].dataset)
        train_acc = train_acc/len(loader['train'].dataset)
        valid_acc = valid_acc/len(loader['valid'].dataset)

        train_result = custom_cm(train_Confusion_matrix)
        valid_result = custom_cm(valid_Confusion_matrix)

        train_recall_list.append(train_result[0])
        train_precision_list.append(train_result[1])
        train_f1score_list.append(train_result[2])

        valid_recall_list.append(valid_result[0])
        valid_precision_list.append(valid_result[1])
        valid_f1score_list.append(valid_result[2])

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_accuracy_list.append(train_acc)
        valid_accuracy_list.append(valid_acc)

        ####################################
        # log all the metrics
        #   compare the performance
        #   against the highest performance
        #   and save the model if its the
        #   highest performance
        #####################################

        Epoch_time = time.time() - start
        Epoch_start = time.strftime("%Y-%m-%d -- %H:%M:%S", time.localtime(start))

        log_name = [
            'Training Accuracy', 'Training Loss', 'Training Recall',
            'Training precision', 'Training F1 Score',
            'Validation Accuracy', 'Validation Loss',  'Validation Recall',
            'Validation precision', 'Validation F1 Score',
        ]

        log_value = [
            train_acc, train_loss, train_result[0], train_result[1], train_result[2],
            valid_acc, valid_loss, valid_result[0], valid_result[1], valid_result[2]
        ]

        logger.log(f"{'Epoch':<25s} --> {epoch}")
        logger.log(f"{'Epoch_Start':<25s} --> {Epoch_start}")
        logger.log(f"{'Epoch_time':<25s} --> {float(Epoch_time):.6}")

        for i, x in enumerate(log_value):
            logger.log(f"{log_name[i]:<25s} --> {float(x):.6}")

        metric_dict = {
            'Validation Accuracy': ['valid_accuracy_max', valid_acc],
            'Validation Recall': ['valid_recall_max', valid_result[0]],
            'Validation Precision': ['valid_precision_max', valid_result[1]],
            'Validation F1 Score': ['valid_f1score_max', valid_result[2]]
        }

        logger.log('\nSaving model:')
        logger.log('-'*15)

        for met in metric_dict:

            valid_met_max = metric_max[metric_dict[met][0]]
            valid_met = metric_dict[met][1]
            save_name = metric_dict[met][0].split("_", 1)[-1]

            if valid_met >= valid_met_max:

                logger.log(
                    f'{met:<25s} --> increased of {float(valid_met-valid_met_max):.6f} '
                    f'({float(valid_met_max):.6f} --> {float(valid_met):.6f})'
                    f' @ Epoch {str(epoch).zfill(3)}'
                )

                metric_max_at_epoch[met] = epoch

                torch.save(
                    model.state_dict(),
                    f'{model_save_path}\\model_{model_name}_{save_name}.pt'
                )
                metric_max[metric_dict[met][0]] = valid_met

        logger.log('\n--------------------------------------------------------------\n')


    ####################################
    # save the performance metrics for
    #   all epochs as a json file
    #####################################

    train_loss_list     = [i.cpu().tolist() for i in train_loss_list]
    valid_loss_list     = [i.cpu().tolist() for i in valid_loss_list]
    train_accuracy_list = [i.cpu().tolist() for i in train_accuracy_list]
    valid_accuracy_list = [i.cpu().tolist() for i in valid_accuracy_list]

    out_json = {
        'train_accuracy_list': train_accuracy_list,
        'train_loss_list': train_loss_list,
        'train_recall_list': train_recall_list,
        'train_precision_list': train_precision_list,
        'train_f1score_list': train_f1score_list,
        'valid_accuracy_list': valid_accuracy_list,
        'valid_loss_list': valid_loss_list,
        'valid_recall_list': valid_recall_list,
        'valid_precision_list': valid_precision_list,
        'valid_f1score_list': valid_f1score_list
    }

    for x in out_json:
        with open(f"{score_save_path}\\model_{model_name}_{x}.json", 'w') as f:
            json.dump(out_json[x], f, indent=2)

    temp_log = [f"\n\nMaximum Valudation Metrics at:", "-"*45]
    for met in metric_max_at_epoch:
        temp = f"Maximum {met:<22s} @ Epoch {metric_max_at_epoch[met]}"
        temp_log.append(temp)
    temp_log.append("-"*45)
    logger.log("\n".join(temp_log))

    return model


def plot_training_graphs(timestamp, SELECTED_MODEL):

    model_name_pre = f"model_{timestamp.replace('-','')}_{SELECTED_MODEL}"
    result_root = f"output\\{timestamp}\\score\\{model_name_pre}"
    out_root = f"output\\{timestamp}\\results"
    if not os.path.exists(out_root): os.makedirs(out_root)

    all_metrics = ['Loss', 'Accuracy', 'Recall', 'Precision', 'F1 Score']

    for a_metric in all_metrics:

        amet = a_metric.lower().replace(' ', '')

        leg_loc = 'lower'
        if amet == 'loss': leg_loc = 'upper'

        with open(f'{result_root}_train_{amet}_list.json', 'r') as r:
            metric_train = json.load(r)
        with open(f'{result_root}_valid_{amet}_list.json', 'r') as r:
            metric_valid = json.load(r)

        plt.ion()
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(metric_train, label=f"train_{amet}")
        plt.plot(metric_valid, label=f"valid_{amet}")
        plt.title(f"Compare Training and Validation {a_metric}")
        plt.xlabel("Epoch #")
        plt.ylabel(a_metric)
        plt.legend(loc=leg_loc+' right')
        plt.savefig(f'{out_root}\\{model_name_pre}_{amet}.png')
        plt.close()
