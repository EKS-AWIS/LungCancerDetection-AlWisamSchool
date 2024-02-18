import os
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay



def predict(image, model, class_name, image_transform):

    try:
        image = image_transform(image)[:3, :, :].unsqueeze(0)
    except:
        image = image.convert('RGB')
        image = image_transform(image)[:3, :, :].unsqueeze(0)

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        model.eval()
        pred = model(image)

    idx = torch.argmax(pred)

    prob = round(pred[0][idx].item()*100, 7)

    return prob, class_name[idx]


def test_model(model, dataset_path, logger, all_classes, image_transform):

    true_count = 0
    all_data   = 0

    root = f"{dataset_path}\\test"

    files = []
    for file_root, _, _files in os.walk(root):
        for afile in _files:
            file_path = os.path.join(file_root, afile)
            all_data += 1

            img = Image.open(file_path)

            class_name = file_path[:-4].rsplit("\\")[-2]
            file_name = file_path[:-4].rsplit("\\")[-1]

            prob, prediction = predict(img, model, all_classes, image_transform)

            logger.log(f"{'File ID':<15s} --> {file_name}")
            logger.log(f"{'File Path':<15s} --> {file_path}")
            logger.log(f"{'True Class':<15s} --> {class_name}")
            logger.log(f"{'Prediction':<15s} --> {prediction}")
            logger.log(f"{'Probability':<15s} --> {prob:.6f}")
            logger.log(f"\n###---{file_name},{class_name},{prediction},{prob}")
            logger.log(f"\n--------------------------------------------------------------\n")

            class_name = file_path.split("\\")[-2]
            if class_name == prediction:
                true_count += 1

    acc = true_count/all_data
    logger.log(f'\n\n\nTest Set Accuracy ===> {acc:.6f}')


def result_processing(timestamp, model_path, test_log_path, all_classes):

    results_out = f'{model_path}\\results'
    if not os.path.exists(results_out):
        os.makedirs(results_out)

    with open(test_log_path, 'r') as r:
        test_logs = r.read()

    test_logs = [
        x.split("###---")[-1].split(",")
        for x in test_logs.split("\n")
        if x.startswith("###")
    ]

    df = pd.DataFrame(
        test_logs,
        columns=['ID', 'True Label', 'Predicted Label', 'Probability']
    )

    #############################################
    # classification report
    #############################################

    report = classification_report(
        df['True Label'],
        df['Predicted Label'],
        target_names=all_classes,
        output_dict=True
    )

    rep_df = pd.DataFrame(report)
    for x in rep_df: rep_df[x] = rep_df[x].map(lambda y: round(y, 4))
    rep_df.loc[['precision', 'recall', 'support'], 'accuracy'] = '-'
    rep_df = rep_df.T.reset_index().rename(columns={'index': 'labels'})

    rep_df.to_excel(f"{results_out}\\Classification_Report_{timestamp}.xlsx", index=False)

    #############################################
    # confusion matrix
    #############################################

    plt.ioff()

    cm = confusion_matrix(
        df['True Label'],
        df['Predicted Label'],
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=all_classes,
    )

    fig, ax = plt.subplots(figsize=(10,10))
    plt.grid(False)
    disp.plot(ax=ax)

    disp.figure_.savefig(
        f"{results_out}\\Confusion_Matrix_{timestamp}.png",
        dpi=1600,
        bbox_inches="tight",
    )

    plt.close()
