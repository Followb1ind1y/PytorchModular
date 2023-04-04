"""
Contains functions to make predictions and evaluations.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_model(model: torch.nn.Module, 
                   dataloaders: torch.utils.data.DataLoader,
                   device: torch.device):
    """
    Evaluate model performance on testset. Return a dictionary contains evaluation information.

    Args:
        model: A target PyTorch model to save.
        dataloaders: A dictionary of DataLoaders for the model to be trained on (dataloaders['test']).
        device: A target device to compute on (e.g. "cuda" or "cpu").
    
    Returns:
            A dictionary of training and testing loss as well as training and testing accuracy metrics. 
            Each metric has a value in a list for each epoch. 
            
            Example return:
                {'labels_list': tensor([1, 1, 0, 1, ...]),
                 'pred_labels_list': tensor([1, 1, 1, 1, ...],
                 'probs_list': tensor([[1.1004e-02, 9.8900e-01],
                                       [2.9027e-02, 9.7097e-01],
                                       [1.3673e-02, 9.8633e-01],
                                       [2.6124e-03, 9.9739e-01],
                                        ...]),
                 'test_acc': [0.9388888888888889]}

    Example usage:
        test_results = evaluate_model(model_ft, dataloaders, device)
    """
    model.eval()
    model.to(device)

    results = {"labels_list": [],
               "pred_labels_list": [],
               "probs_list": [],
               "test_acc": []
               }

    running_accs = []

    batch_iter = tqdm(
        enumerate(dataloaders['val']),
        total=len(dataloaders['val']),
    )

    for i, (x, y) in batch_iter:
        batch_iter.set_description(f'Test Step {i}')
        # Send to device (GPU or CPU)
        inputs = x.to(device)
        labels = y.to(device)

        with torch.no_grad():
            # Forward pass
            outputs = model(inputs)

            y_prob = torch.softmax(outputs, dim=1)
            y_pred_label = torch.argmax(y_prob, dim=1)

            results['labels_list'].append(labels.cpu())
            results['pred_labels_list'].append(y_pred_label.cpu())
            results['probs_list'].append(y_prob.cpu())

            acc_value = (y_pred_label == labels).sum().item()/len(outputs)
            running_accs.append(acc_value)

    results['labels_list'] = torch.cat(results['labels_list'], dim=0)
    results['pred_labels_list'] = torch.cat(results['pred_labels_list'], dim=0)
    results['probs_list'] = torch.cat(results['probs_list'], dim=0)
    results["test_acc"].append(np.mean(running_accs))

    return results

# Plot the confusion matrix
def plot_confusion_matrix(labels, pred_labels, class_names):
    """
    Plot the confusion matrix base on the test results.

    Args:
        labels: A list contains all the true labels.
        pred_labels: A list contains all the predicted labels.
        class_names: A list contains all the classes (e.g. ['ants', 'bees']).
    """
    
    fig = plt.figure(figsize = (8, 8));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels = class_names);
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    fig.delaxes(fig.axes[1]) #delete colorbar
    plt.xticks(rotation = 90)
    plt.xlabel('Predicted Label', fontsize = 10)
    plt.ylabel('True Label', fontsize = 10)