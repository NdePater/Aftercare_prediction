import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
from torch.autograd import Variable, Function
from torch import autograd
from helpers import EarlyStopper, plot_training_progress, get_confusion_matrix, calculate_metrics, show_confusion_matrix
from sklearn.metrics import accuracy_score
import wandb
# from predict import get_predictions
import torch.profiler as profiler
from predict import predict, get_val_predictions

m = torch.nn.Sigmoid()

class Train():
    """
    A class to handle the training process of a neural network model.
    
    Attributes:
        model (torch.nn.Module): The neural network model to be trained.
        handler (CustomDataHandler): Data handler for training data.
        val_handler (CustomDataHandler): Data handler for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        num_epochs (int): Number of epochs for training.
        use_wandb (bool): Flag to use Weights and Biases for logging.
        show_visuals (bool): Flag to show training visuals.
        early_stopper (EarlyStopper): Early stopping mechanism.
        device (torch.device): Device to run the training on (CPU or GPU).
    """
    def __init__(self, model, handler, val_handler, optimizer, criterion, num_epochs, use_wandb, show_visuals, early_stopper=None):
        """
        Initializes the Train class with the given parameters.
        
        Args:
            model (torch.nn.Module): The neural network model to be trained.
            handler (CustomDataHandler): Data handler for training data.
            val_handler (CustomDataHandler): Data handler for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (torch.nn.Module): Loss function.
            num_epochs (int): Number of epochs for training.
            use_wandb (bool): Flag to use Weights and Biases for logging.
            show_visuals (bool): Flag to show training visuals.
            early_stopper (EarlyStopper, optional): Early stopping mechanism. Defaults to None.
        """
        self.model          = model
        self.handler        = handler
        self.val_handler    = val_handler
        self.optimizer      = optimizer
        self.num_epochs     = num_epochs 
        self.use_wandb      = use_wandb
        self.show_visuals   = show_visuals

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model      = model.to(device)
        self.criterion  = criterion.to(device)
        
        self.early_stopper = EarlyStopper(patience=5, min_delta=0)

    def log_wandb(self, train_loss, val_loss, val_accuracy, accuracy_filtered, metrics):
        """
        Logs the training and validation metrics to Weights and Biases.
        
        Args:
            train_loss (float): Training loss.
            val_loss (float): Validation loss.
            val_accuracy (float): Validation accuracy.
            accuracy_filtered (float): Filtered validation accuracy.
            metrics (dict): Additional metrics to log.
        """
        if self.use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy, "val_min_accuracy": accuracy_filtered})
            wandb.log(metrics)

    def early_stop(self, epoch, val_loss):
        """
        Checks if early stopping should be triggered.
        
        Args:
            epoch (int): Current epoch number.
            val_loss (float): Validation loss.
        
        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.early_stopper is not None:
            # In the very early epochs sometimes the early stopping triggers to early
            if epoch> 20 and self.early_stopper.early_stop(val_loss):
                print("Early stopping triggered")
                return True

    def show_output(self, train_losses, val_losses, val_errors, val_min_errors, epoch, num_epochs):
        """
        Displays the training progress.
        
        Args:
            train_losses (list): List of training losses.
            val_losses (list): List of validation losses.
            val_errors (list): List of validation errors.
            val_min_errors (list): List of validation minority errors.
            epoch (int): Current epoch number.
            num_epochs (int): Total number of epochs.
        """
        if self.show_visuals:
            plot_training_progress(train_losses, val_losses, val_errors, val_min_errors)
        else:
            message = f"Epoch [{epoch+1}/{num_epochs}], Train_loss: {train_losses[-1]} Val_Loss: {val_losses[-1]}, Val Accuracy: {val_errors[-1]} Val minority {val_min_errors[-1]}"
            tqdm.write(message)

    def train(self, use_profiler):
        """
        Trains the model for the specified number of epochs.
        
        Args:
            use_profiler (bool): Flag to use profiler for performance analysis.
        """
        train_losses, val_losses, val_errors, val_min_errors    = [],[],[],[]

        # Define the profiler
        if use_profiler:
            prof = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=1,  # Number of warm-up steps
                warmup=1,
                active=5,  # Number of steps to actually profile
                repeat=1,  # Repeat profiling cycle for more data
            ),
            on_trace_ready=profiler.tensorboard_trace_handler('./runs'),  # For TensorBoard
            record_shapes=True,
            with_stack=True)
            prof.start()
        
        for epoch in tqdm(range(self.num_epochs)):
            if use_profiler:
                prof.step()
            
            train_loss  = self.train_epoch()
            val_loss    = self.evaluate()

            val_min_acc, val_maj_acc = self.get_accuracies(self.val_handler)

            # get the validation predictions
            predictions         = get_val_predictions(self.model, self.handler)
            
            metrics             = calculate_metrics(predictions)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_errors.append(1 - val_maj_acc)
            val_min_errors.append(1 - val_min_acc)

            self.log_wandb(train_loss, val_loss, val_maj_acc, val_min_acc, metrics)
            
            if self.early_stop(epoch, val_loss): break                

            # if epoch % 5 == 0: self.show_output(train_losses, val_losses, val_errors, val_min_errors, epoch, self.num_epochs)

        if use_profiler:
            prof.stop()
    

    def train_epoch(self):
        """
        Trains the model for one epoch.
        
        Returns:
            float: Average training loss for the epoch.
        """
        # with autograd.detect_anomaly():
        self.model.train()
        epoch_loss = 0.0
        for inputs, labels in self.handler.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            m_outputs = m(outputs)

            loss = self.criterion(m_outputs, labels)
            loss.backward()
            # TODO Check if this is still necessary
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            # check_gradients(model)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.handler.train_loader)

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        # val_predictions = []

        val_epoch_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in self.val_handler.val_loader:
                val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                
                val_outputs = self.model(val_inputs)
                m_val_outputs = m(val_outputs)

                val_loss = self.criterion(m_val_outputs, val_labels)
                val_epoch_loss += val_loss.item()

        return val_epoch_loss / len(self.val_handler.val_loader)

    def get_accuracies(self, handler):
        """
        Calculates the accuracies for the majority and minority classes.
        
        Args:
            handler (CustomDataHandler): Data handler for validation data.
        
        Returns:
            tuple: Majority and minority class accuracies.
        """
        val_pred = predict(self.model, handler.X_val_tensor)

        val_pred_min = val_pred[ handler.y_val_min_mask]
        val_pred_maj = val_pred[~handler.y_val_min_mask]

        val_maj_acc = accuracy_score(handler.y_val_maj,  val_pred_maj)
        val_min_acc = accuracy_score(handler.y_val_min,  val_pred_min)

        return val_maj_acc, val_min_acc

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the given dataloader.
    
    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
    
    Returns:
        float: Average validation loss.
    """
    model.eval()
    # val_predictions = []

    val_epoch_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in dataloader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            # _, val_predicted = torch.max(val_outputs.data, 1)
            # val_predictions.extend(val_predicted.tolist())
            m_val_outputs = m(val_outputs)
            val_loss = criterion(m_val_outputs, val_labels)
            val_epoch_loss += val_loss.item()

    return val_epoch_loss / len(dataloader)
