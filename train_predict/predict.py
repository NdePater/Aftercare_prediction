from IPython.display import display
import torch
import pandas as pd
import numpy as np

def predict(model, xtensor, threshold=0.5):
    """
    Predicts the labels for the given input tensor using the specified threshold.
    
    Args:
        model (torch.nn.Module): The neural network model to be used for prediction.
        xtensor (torch.Tensor): Input tensor.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.
    
    Returns:
        np.ndarray: Predicted labels.
    """
    # get the raw prediction from the model
    test_predictions = predict_raw(model, xtensor)
    # compare it to the threshold
    test_predictions = np.where(test_predictions > threshold, 1, 0)
    return test_predictions

def predict_raw(model, xtensor):
    """
    Gets the raw predictions from the model.
    
    Args:
        model (torch.nn.Module): The neural network model to be used for prediction.
        xtensor (torch.Tensor): Input tensor.
    
    Returns:
        np.ndarray: Raw predictions.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_outputs = model(xtensor.to(device))
    # map the outputs to [0, 1]
    test_predictions = torch.sigmoid(test_outputs).cpu().detach().numpy()

    test_predictions = test_predictions.squeeze()

    return test_predictions

def get_test_predictions(model, handler, treshold=0.5):
    """
    Gets the test predictions and creates a DataFrame for exporting.
    
    Args:
        model (torch.nn.Module): The neural network model to be used for prediction.
        handler (CustomDataHandler): Data handler for test data.
        treshold (float, optional): Threshold for binary classification. Defaults to 0.5.
    
    Returns:
        pd.DataFrame: DataFrame containing test predictions and ground truth labels.
    """
    test_predictions = predict(model, handler.X_test_tensor, treshold)

    # create a df for exporting
    predictions_df = pd.DataFrame(columns= ['Patient ID'] + handler.features + ['Predicted Label', 'Ground Truth Label'])

    #add the x_tensor to the df
    predictions_df['Patient ID'] = handler.id_test
    predictions_df[handler.features] = handler.get_inverse_scaler(handler.X_test)
    predictions_df['Predicted Label'] = test_predictions.tolist()
    predictions_df['Ground Truth Label'] = handler.y_test.tolist()

    return predictions_df

def get_val_predictions(model, handler):
    """
    Gets the validation predictions and creates a DataFrame for exporting.
    
    Args:
        model (torch.nn.Module): The neural network model to be used for prediction.
        handler (CustomDataHandler): Data handler for validation data.
    
    Returns:
        pd.DataFrame: DataFrame containing validation predictions and ground truth labels.
    """
    val_predictions = predict(model, handler.X_val_tensor)

    # create a df for exporting
    predictions_df = pd.DataFrame(columns= handler.features + ['Predicted Label', 'Ground Truth Label'])

    # If ID val is fixed, uncomment the line below, and add the ID to the dataframe columns
    # predictions_df['Patient ID'] = handler.id_val
    predictions_df[handler.features] = handler.get_inverse_scaler(handler.X_val)
    predictions_df['Predicted Label'] = val_predictions.tolist()
    predictions_df['Ground Truth Label'] = handler.y_val.tolist()

    return predictions_df
  

def show_predictions(prediction_df, amount):
    """
    Displays the first few entries of true positives, false negatives, false positives, and true negatives.
    
    Args:
        prediction_df (pd.DataFrame): DataFrame containing predictions and ground truth labels.
        amount (int): Number of entries to display.
    """
    FP = prediction_df[(prediction_df['Predicted Label'] == 1) & (prediction_df['Ground Truth Label'] == 0)]
    FN = prediction_df[(prediction_df['Predicted Label'] == 0) & (prediction_df['Ground Truth Label'] == 1)]
    TP = prediction_df[(prediction_df['Predicted Label'] == 1) & (prediction_df['Ground Truth Label'] == 1)]
    TN = prediction_df[(prediction_df['Predicted Label'] == 0) & (prediction_df['Ground Truth Label'] == 0)]

    for name, subset in [("TP", TP), ("FN", FN), ("FP", FP), ("TN", TN)]:
        print(f"Showing the first {amount} entries classified as {name} =====================")
        display(subset.head(amount))
