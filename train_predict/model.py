import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    """
    A simple neural network model with one hidden layer.
    
    Attributes:
        hidden_size (int): Size of the hidden layer.
        fc1 (torch.nn.Linear): First fully connected layer.
        relu (torch.nn.LeakyReLU): Leaky ReLU activation function.
        dropout (torch.nn.Dropout): Dropout layer.
        fc2 (torch.nn.Linear): Second fully connected layer.
        weight_decay (float): Weight decay for L2 regularization.
    """
    def __init__(self, input_size, num_classes, hidden_size, weight_decay, drop=0.5):
        """
        Initializes the NeuralNetwork class with the given parameters.
        
        Args:
            input_size (int): Size of the input layer.
            num_classes (int): Number of output classes.
            hidden_size (int): Size of the hidden layer.
            weight_decay (float): Weight decay for L2 regularization.
            drop (float, optional): Dropout rate. Defaults to 0.5.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.weight_decay = weight_decay

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def l2_regularization(self):
        """
        Calculates the L2 regularization term.
        
        Returns:
            torch.Tensor: L2 regularization term.
        """
        l2_reg = torch.tensor(0.0)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.weight_decay * l2_reg

class FlexibleNeuralNetwork(nn.Module):
    """
    A flexible neural network model with multiple hidden layers.
    
    Attributes:
        hidden_size (int): Size of the hidden layers.
        fc2 (torch.nn.Linear): Output fully connected layer.
        weight_decay (float): Weight decay for L2 regularization.
        model (torch.nn.Sequential): Sequential model containing hidden layers.
    """
    def __init__(self, input_size, num_classes, hidden_size, hidden_layers, weight_decay, drop=0.5):
        """
        Initializes the FlexibleNeuralNetwork class with the given parameters.
        
        Args:
            input_size (int): Size of the input layer.
            num_classes (int): Number of output classes.
            hidden_size (int): Size of the hidden layers.
            hidden_layers (int): Number of hidden layers.
            weight_decay (float): Weight decay for L2 regularization.
            drop (float, optional): Dropout rate. Defaults to 0.5.
        """
        super().__init__()
        layers = []
        prev_size = input_size
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(drop))
            prev_size = hidden_size
        
        self.hidden_size = hidden_size
        self.fc2 = nn.Linear(prev_size, num_classes)
        self.weight_decay = weight_decay
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        model_out = self.model(x)
        out = self.fc2(model_out)
        return out

    def l2_regularization(self):
        """
        Calculates the L2 regularization term.
        
        Returns:
            torch.Tensor: L2 regularization term.
        """
        l2_reg = torch.tensor(0.0)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.weight_decay * l2_reg

class SideTune(nn.Module):
    """
    A model that combines the outputs of a base model and a side model using a weighted sum.
    
    Attributes:
        base_model (torch.nn.Module): The base model.
        side_model (torch.nn.Module): The side model.
        alpha (torch.nn.Parameter): Weighting parameter for combining the outputs.
    """
    def __init__(self, base_model, side_model, alpha):
        """
        Initializes the SideTune class with the given parameters.
        
        Args:
            base_model (torch.nn.Module): The base model.
            side_model (torch.nn.Module): The side model.
            alpha (float): Initial value for the weighting parameter.
        """
        super().__init__()
        self.base_model = base_model
        self.side_model = side_model
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        """
        Defines the forward pass of the SideTune model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        base_out = self.base_model(x)
        side_out = self.side_model(x)
        output = self.alpha * base_out + (1 - self.alpha) * side_out
        return output

class SideTuneWrapper():
    """
    A wrapper class for the SideTune model.
    
    Attributes:
        base_model (torch.nn.Module): The base model.
        side_model (torch.nn.Module): The side model.
        alpha (float): Initial value for the weighting parameter.
        side_tuning_network (SideTune): The SideTune model.
    """
    def __init__(self, base_model, side_model, alpha):
        """
        Initializes the SideTuneWrapper class with the given parameters.
        
        Args:
            base_model (torch.nn.Module): The base model.
            side_model (torch.nn.Module): The side model.
            alpha (float): Initial value for the weighting parameter.
        """
        self.base_model = base_model
        self.side_model = side_model
        self.alpha = alpha
        self.side_tuning_network = SideTune(self.base_model, self.side_model, self.alpha)

