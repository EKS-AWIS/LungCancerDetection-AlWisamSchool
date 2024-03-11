import os
import time
import timm

import torch
from torchvision.transforms import transforms

from DatasetLoader import load_dataset
from Training import train_model, plot_training_graphs
from Testing import test_model, result_processing

import argparse

import warnings
warnings.filterwarnings("ignore")

########################################################################
# define global variables
########################################################################

class Logger:
    """ class to handle logging training and testing information """
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, 'w') as w:
            pass

    def log(self, log):
        print(log)
        with open(self.log_path, 'a') as w:
            w.write(log)
            w.write("\n")

    def get_log_path(self):
        return self.log_path


AVAILABLE_CLASSES_LIST = [
    [ 'Malignant', 'Non Malignant' ],
    [ 'Bengin', 'Malignant', 'Normal' ]
]


IMAGE_PREPROCESSING = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

FULLY_CONNECTED_LAYER = torch.nn.Sequential(
    torch.nn.Linear(2048, 256),
    torch.nn.Dropout(0.2),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 64),
    torch.nn.Dropout(0.2),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.Dropout(0.2),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 4),
    torch.nn.Softmax(dim=1)
)


def run_main(
        var_selected_model,
        var_transfer_learning,
        var_n_epochs,
        var_batch_size,
        var_lr,
        var_wd,
        var_class=0,
    ):

    ########################################################################
    # define paths
    ########################################################################

    #ANCHOR - define class names
    AVAILABLE_CLASSES = AVAILABLE_CLASSES_LIST[var_class]

    # define dataset file path
    Dataset = '.\\dataset'

    # define unique id for output path using timestamp
    timestamp = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
    timestamp_short = timestamp.replace('-','')

    # create output path
    model_path = f'output\\{timestamp}'
    if not os.path.exists(model_path): os.makedirs(model_path)
    print(f"\t{model_path}\n")

    ########################################################################
    # load data
    ########################################################################

    #ANCHOR - setting data hyperparameters
    batch_size = var_batch_size
    num_workers = 0
    shuffle = True

    # load dataset
    train_loader, valid_loader = load_dataset(
        Dataset, IMAGE_PREPROCESSING,
        batch_size, num_workers, shuffle
    )
    loaders = { 'train': train_loader, 'valid': valid_loader }

    ########################################################################
    # initate model
    ########################################################################

    #ANCHOR - initiate model
    SELECTED_MODEL = var_selected_model
    model = timm.create_model(SELECTED_MODEL, pretrained=True)

    # check if GPU is available
    use_cuda = torch.cuda.is_available()
    if use_cuda: model = model.cuda()

    #ANCHOR - to freeze pretrained weights or not
    transfer_learning = var_transfer_learning
    for param in model.parameters():
        param.requires_grad = transfer_learning

    # define fully connected layer and unfreeze weights
    model.fc = FULLY_CONNECTED_LAYER
    for param in model.fc.parameters():
        param.requires_grad = True

    # check if GPU is available
    if use_cuda: model_transfer = model.cuda()
    else: model_transfer = model

    ########################################################################
    # define hyperparameters
    ########################################################################

    #ANCHOR - define hyperparameters
    n_epochs         = var_n_epochs
    lr               = var_lr
    wd               = var_wd
    model_name       = f'{timestamp.replace("-","")}_{SELECTED_MODEL}'
    save_model_path  = model_path

    # initiate the loss function and the optimization function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model_transfer.fc.parameters(),
        lr=lr,
        weight_decay=wd
    )

    # define logger class to log training
    logger = Logger(f'{model_path}\\train_log.txt')

    # log all defined hyperparameters
    logger.log("-"*60)
    logger.log('Model Parameters:')
    logger.log("-"*60)
    logger.log(f"- {'Model':<30s} --> {SELECTED_MODEL}")
    logger.log(f"- {'No. of Epochs':<30s} --> {n_epochs}")
    logger.log(f"- {'Learning Rate':<30s} --> {lr}")
    logger.log(f"- {'Weight Decay':<30s} --> {wd}")
    logger.log(f"- {'Batch Size':<30s} --> {batch_size}")
    logger.log(f"- {'Loss Function':<30s} --> {str(criterion)}")
    logger.log(f"- {'Optimization Function':<30s} --> {str(optimizer).split()[0]}")
    logger.log(f"- {'Freeze Pretrained Weights':<30s} --> {transfer_learning}")
    logger.log("-"*60)
    logger.log("")

    ########################################################################
    # start training and plot graphs for the training
    ########################################################################

    # Call train function with all the parameters
    model = train_model(
        Dataset, n_epochs, loaders, model,
        optimizer, criterion, use_cuda,
        save_model_path, model_name, logger
    )

    # evaluate model
    plot_training_graphs(timestamp, SELECTED_MODEL)

    ########################################################################
    # start testing the model on the test dataset (unseen data)
    ########################################################################

    # get pretrained model base
    model_test = timm.create_model(SELECTED_MODEL, pretrained=True)

    # get fully connected layer
    model_test.fc = FULLY_CONNECTED_LAYER

    # load trained model
    trained_model_path = (
        f"output\\{timestamp}\\models\\"
        f"model_{timestamp_short}_{SELECTED_MODEL}_f1score_max.pt"
    )
    model_test.load_state_dict(torch.load(trained_model_path))

    # check if GPU is available
    use_cuda = torch.cuda.is_available()
    if use_cuda: model_test = model_test.cuda()

    # define logger class to log testing
    test_log = f'{model_path}\\test_log.txt'
    logger = Logger(test_log)

    # Call test function with all the parameters
    test_model(
        model_test, Dataset, logger,
        AVAILABLE_CLASSES, IMAGE_PREPROCESSING
    )

    # evaluate model by creating classifcation report and confusion matrix
    result_processing(timestamp_short, model_path, test_log, AVAILABLE_CLASSES)

    ########################################################################
    # end
    ########################################################################

    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m' , '--Model', type=str, default='resnet50')
    parser.add_argument('-tl', '--TransferLearn', type=bool, choices=[True, False], default=False)
    parser.add_argument('-e' , '--Epochs', type=int, default=2)
    parser.add_argument('-b',  '--BatchSize', type=int, default=32)
    parser.add_argument('-lr', '--LearningRate', type=float, default=0.001)
    parser.add_argument('-wd', '--WeightDecay', type=float, default=0.005)
    parser.add_argument('-c' , '--Classes', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = list(vars(parse_arguments()).values())
    run_main(*args)
