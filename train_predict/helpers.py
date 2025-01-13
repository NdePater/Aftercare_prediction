import seaborn as sns
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.autograd import Variable
from torch import autograd
# from dataset import CustomDataHandler, CustomDataset
from predict import show_predictions, predict, predict_raw
from sklearn.metrics import accuracy_score
from torch import Tensor
from sklearn.metrics import precision_recall_curve, auc
from predict import get_val_predictions, get_test_predictions

def generate_test_results(test_predictions):
    """
    Generate test results including test accuracy and accuracy filtered by minority class.

    Args:
        test_predictions (pd.DataFrame): DataFrame containing predicted and ground truth labels.

    Returns:
        tuple: Test accuracy and accuracy filtered by minority class.
    """
    y_pred = test_predictions['Predicted Label']
    y_true = test_predictions['Ground Truth Label']
    y_true_min_mask = y_true == 1

    y_true_min = y_true[y_true_min_mask]
    y_pred_min = y_pred[y_true_min_mask]

    test_accuracy       = accuracy_score(y_true,      y_pred    )  
    accuracy_filtered   = accuracy_score(y_true_min,  y_pred_min)

    return test_accuracy, accuracy_filtered

def one_hot(index, classes):
    """
    Convert class indices to one-hot encoded vectors.

    Args:
        index (torch.Tensor): Tensor containing class indices.
        classes (int): Number of classes.

    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    device = index.device  # Get the device of the input tensor (GPU/CPU)
    size = index.size(0)  # Get the batch size (number of observations)
    
    # Create a mask of zeros with shape (batch_size, num_classes)
    mask = torch.zeros(size, classes, device=device)
    
    # Ensure index is reshaped correctly: (batch_size, 1)
    index = index.view(-1, 1)
    
    # Scatter 1s in the mask based on the index
    mask.scatter_(1, index, 1.0)
    
    return mask

def calculate_focal_alpha(handler):
    """
    Calculate the focal alpha value for imbalanced datasets.

    Args:
        handler (CustomDataHandler): Data handler containing training data.

    Returns:
        float: Focal alpha value.
    """
    targets       = handler.y_train
    class_counts  = targets.value_counts()
    alpha = 1.0 / (class_counts + 1e-7)
    alpha = alpha / alpha.sum()
    print(f"suggested alpha is {alpha}")
    alpha = float(alpha[1])
    print(f"alpha {alpha}")
    # TODO: check if this is correct
    # alpha = 1 - 2*float(alpha[0])
    return alpha

class EarlyStopper:
    """
    Early stopping mechanism to prevent overfitting during training.
    """
    def __init__(self, patience=1, min_delta=0):
        """
        Initialize the EarlyStopper.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Check if training should be stopped early.

        Args:
            validation_loss (float): Current validation loss.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_training_progress(train_losses, val_losses, val_errors, val_min_errors):
    """
    Plot the training progress including training and validation losses and errors.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        val_errors (list): List of validation errors.
        val_min_errors (list): List of minimum validation errors.
    """
    clear_output(wait=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, label='Training Loss', color=color)
    ax1.plot(val_losses, label='Validation Loss', color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(min(train_losses+val_losses), max(train_losses+val_losses))
    # ax1.set_ylim(1000, 2500)
    # plot it on the center right, to make sure they are not same heights
    plt.legend(loc = 'center right')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Error', color=color)
    ax2.plot(val_errors, label='Validation Error', color=color)
    ax2.plot(val_min_errors, label='Validation Min Error', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    plt.title('Training Progress')
    plt.legend(loc = 'upper right')
    plt.show()

def check_gradients(model):
    """
    Check gradients of the model parameters for NaNs or Infs.

    Args:
        model (torch.nn.Module): PyTorch model.
    """
    for param in model.parameters():
        if param.grad is not None:
            print("gradient")
            print(param.grad)
            if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                print("Bad gradient detected!")
                print(param.grad)
                print(param)

def calculate_metrics(prediction_df):
    """
    Calculate various classification metrics.

    Args:
        prediction_df (pd.DataFrame): DataFrame containing predicted and ground truth labels.

    Returns:
        dict: Dictionary containing accuracy, recall, specificity, precision, npv, and f1_score.
    """
    # Calculate metrics
    TP = len(prediction_df[(prediction_df['Predicted Label'] == 1) & (prediction_df['Ground Truth Label'] == 1)])
    TN = len(prediction_df[(prediction_df['Predicted Label'] == 0) & (prediction_df['Ground Truth Label'] == 0)])
    FP = len(prediction_df[(prediction_df['Predicted Label'] == 1) & (prediction_df['Ground Truth Label'] == 0)])
    FN = len(prediction_df[(prediction_df['Predicted Label'] == 0) & (prediction_df['Ground Truth Label'] == 1)])
    
    try:
        accuracy    = (TP + TN) / (TP + TN + FP + FN)
        recall      = TP        / (TP + FN)
        specificity = TN        / (TN + FP)
        precision   = TP        / (TP + FP)
        f1_score    = 2 * (precision * recall) / (precision + recall)
        npv         = TN        / (TN + FN)

    except ZeroDivisionError:
        accuracy    = 0
        recall      = 0
        specificity = 0
        precision   = 0
        f1_score    = 0
        npv         = 0

    # f1_score    = 2 * (precision * recall) / (precision + recall)
    
    return {"accuracy": accuracy, "recall": recall, "specificity": specificity, "precision": precision, "npv": npv, "f1_score": f1_score}

def show_confusion_matrix(confusion_matrix):
    """
    Display a confusion matrix as a heatmap.

    Args:
        confusion_matrix (torch.Tensor): Confusion matrix tensor.
    """
    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def get_confusion_matrix(prediction_df):
    """
    Generate a confusion matrix from prediction DataFrame.

    Args:
        prediction_df (pd.DataFrame): DataFrame containing predicted and ground truth labels.

    Returns:
        torch.Tensor: Confusion matrix tensor.
    """
    confusion_matrix = torch.zeros(2, 2)
    FP = prediction_df[(prediction_df['Predicted Label'] == 1) & (prediction_df['Ground Truth Label'] == 0)]
    FN = prediction_df[(prediction_df['Predicted Label'] == 0) & (prediction_df['Ground Truth Label'] == 1)]
    TP = prediction_df[(prediction_df['Predicted Label'] == 1) & (prediction_df['Ground Truth Label'] == 1)]
    TN = prediction_df[(prediction_df['Predicted Label'] == 0) & (prediction_df['Ground Truth Label'] == 0)]

    confusion_matrix[0, 0] = TN.shape[0]
    confusion_matrix[0, 1] = FP.shape[0]
    confusion_matrix[1, 0] = FN.shape[0]
    confusion_matrix[1, 1] = TP.shape[0]

    return confusion_matrix

def generate_test_stats(model, handler, best_threshold, on = 'val'):
    """
    Generate test statistics including accuracy, confusion matrix, and other metrics.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        handler (CustomDataHandler): Data handler containing test data.
        best_threshold (float): Best threshold for classification.
        on (str): Dataset to evaluate on ('val' or 'test').

    Returns:
        tuple: Test accuracy and accuracy filtered by minority class.
    """
    if on == 'val':
        df_prediction = get_val_predictions(model, handler)
    elif on == 'test':
        df_prediction = get_test_predictions(model, handler, best_threshold)

    test_accuracy, accuracy_filtered      = generate_test_results(  df_prediction)
    confusion_matrix                      = get_confusion_matrix(   df_prediction)
    metrics                               = calculate_metrics(      df_prediction)

    show_results(df_prediction, test_accuracy, accuracy_filtered, confusion_matrix, metrics)
    return test_accuracy, accuracy_filtered

def show_results(prediction_df, acc_maj, acc_min, confusion_matrix, metrics):
    """
    Display the results including predictions, accuracy, and confusion matrix.

    Args:
        prediction_df (pd.DataFrame): DataFrame containing predicted and ground truth labels.
        acc_maj (float): Majority class accuracy.
        acc_min (float): Minority class accuracy.
        confusion_matrix (torch.Tensor): Confusion matrix tensor.
        metrics (dict): Dictionary containing various classification metrics.
    """
    # Show the first N predictions of TP, FP, TN, FN
    show_predictions(prediction_df, 5)
    # Show the test accuracy and minority
    print(f"Test Majority Accuracy: {acc_maj}, Test Minority Accuracy: {acc_min}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    show_confusion_matrix(confusion_matrix)

def build_pr_curve(model, handler, debug =False, on = 'val'):
    """
    Build and display the Precision-Recall curve.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        handler (CustomDataHandler): Data handler containing validation or test data.
        debug (bool): If True, return additional debug information.
        on (str): Dataset to evaluate on ('val' or 'test').

    Returns:
        tuple: Best precision, recall, threshold, and PR AUC score.
    """
    # create a plot
    plt.figure(figsize=(6, 6))

    if on == 'val':
        y_true    = handler.y_val_tensor.cpu().numpy()
        y_scores  = predict_raw(model, handler.X_val_tensor)
    elif on == 'test':
        y_true    = handler.y_test_tensor.cpu().numpy()
        y_scores  = predict_raw(model, handler.X_test_tensor)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    pr_auc = auc(recall, precision)
    print(pr_auc)
    # ## cost based thresholding
    # # Define costs
    # c_fp = 2  # Cost of a false positive
    # c_fn = 1  # Cost of a false negative

    # # Calculate False Positives (FP) and False Negatives (FN)
    # fp = (1 - precision) * (precision * recall * len(y_true))
    # fn = (1 - recall) * sum(y_true)

    # # Calculate cost for each threshold
    # costs = c_fp * fp + c_fn * fn
    # best_index = np.argmin(costs)

    ### F-based thresholding
    # beta = 1
    # best_index = np.argmax(((1+beta**2) * precision * recall) / (beta**2 * precision + recall))

    ### 
    best_index = np.argmax(2* (precision * recall)/(precision + recall))

    best_threshold  = thresholds[best_index]
    best_recall     = recall[best_index]
    best_precision  = precision[best_index]

    print(f"from this pr curve, minimizing costs revealed the best threshold of {best_threshold} with a recall of {best_recall} and precision of {best_precision}")
    
    plt.plot(recall, precision)
    # set the ranges to 0,1 0,1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    if debug:
        return precision, recall, thresholds, best_precision, best_recall, best_threshold, pr_auc
    else:
        return best_precision, best_recall, best_threshold, pr_auc