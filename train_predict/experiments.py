import wandb
import torch
import torch.optim as optim

from model import NeuralNetwork, SideTuneWrapper, FlexibleNeuralNetwork
from losses import FocalLoss
from train import Train
from helpers import generate_test_stats, calculate_focal_alpha, build_pr_curve


class Experiment():
    """
    A class to handle the setup and execution of a regular experiment.
    
    Attributes:
        handler (CustomDataHandler): Data handler for training data.
        config (dict): Configuration parameters for the experiment.
        id_number (str): Identifier for the experiment.
        experiment_type (str): Type of the experiment.
        name (str): Name of the experiment.
        device (torch.device): Device to run the experiment on (CPU or GPU).
    """
    def __init__(self, handler, config, i):
        """
        Initializes the Experiment class with the given parameters.
        
        Args:
            handler (CustomDataHandler): Data handler for training data.
            config (dict): Configuration parameters for the experiment.
            i (int): Identifier for the experiment.
        """
        self.handler = handler
        self.config = config
        self.id_number = f"{i}"
        self.experiment_type = f"regular"
        self.name = self.experiment_type + "_" + self.id_number
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    @staticmethod
    def generate_experiment(primary_handler, secondary_handler, config, i):
        """
        Generates an instance of the Experiment class.
        
        Args:
            primary_handler (CustomDataHandler): Primary data handler.
            secondary_handler (CustomDataHandler): Secondary data handler.
            config (dict): Configuration parameters for the experiment.
            i (int): Identifier for the experiment.
        
        Returns:
            Experiment: An instance of the Experiment class.
        """
        if config['apply_smote']:
            primary_handler.apply_smote()
            secondary_handler.apply_smote()
        return Experiment(primary_handler, config, i)

    def start_wandb(self, name):
        """
        Initializes Weights and Biases for logging.
        
        Args:
            name (str): Name of the experiment.
        """
        wandb.init(
            project="thesis_baselines",
            config = self.config,
            name = name,
            group= self.experiment_type
        )
        self.config['type'] = self.experiment_type

    def get_network_shape(self, handler):
        """
        Gets the input and output shape of the network.
        
        Args:
            handler (CustomDataHandler): Data handler.
        
        Returns:
            tuple: Input and output shape of the network.
        """
        return len(handler.features), 1

    def build_model(self, model_type, model_params):
        """
        Builds the neural network model.
        
        Args:
            model_type (str): Type of the model.
            model_params (list): Parameters for the model.
        
        Returns:
            torch.nn.Module: The neural network model.
        """
        if model_type == "NeuralNetwork":
            return NeuralNetwork(*model_params)
        elif model_type =="Flexible":
            return FlexibleNeuralNetwork(*model_params)

    def build_criterion(self, criterion_type, criterion_params):
        """
        Builds the loss function.
        
        Args:
            criterion_type (str): Type of the loss function.
            criterion_params (list): Parameters for the loss function.
        
        Returns:
            torch.nn.Module: The loss function.
        """
        if criterion_type == "FocalLoss":
            return FocalLoss(*criterion_params)

    def build_optimizer(self, optimizer_type, optimizer_params):
        """
        Builds the optimizer.
        
        Args:
            optimizer_type (str): Type of the optimizer.
            optimizer_params (list): Parameters for the optimizer.
        
        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        if optimizer_type == "Adam":
            return optim.Adam(*optimizer_params)

    def run_experiment(self, use_wandb, show_visuals, sweep, use_profiler):
        """
        Runs the experiment.
        
        Args:
            use_wandb (bool): Flag to use Weights and Biases for logging.
            show_visuals (bool): Flag to show training visuals.
            sweep (bool): Flag to indicate if the experiment is part of a sweep.
            use_profiler (bool): Flag to use profiler for performance analysis.
        """
        config = self.config

        if use_wandb and not sweep:
            self.start_wandb(self.name)

        if config['alpha'] is None:
            print("alpha was none")
            new_alpha = calculate_focal_alpha(self.handler)
            config['alpha'] = new_alpha
            print(f"calculated alpha to be {config['alpha']}")

        model_in, model_out = self.get_network_shape(self.handler)
        self.model = self.build_model("Flexible", [model_in, model_out, config['hidden_size'], config['hidden_layers'], config['weight_decay'], config['dropout']])
        self.model.to(self.device)
        
        criterion  = self.build_criterion(config['criterion_type'], [config['alpha'], config['gamma']])
        optimizer  = self.build_optimizer(config['optimizer_type'], [self.model.parameters(), config['lr']])

        criterion.to(self.device)

        train_class = Train(self.model, self.handler, self.handler, optimizer, criterion, config['num_epochs'], use_wandb, show_visuals)
        train_class.train(use_profiler)
                
        best_precision, best_recall, best_threshold, pr_auc = build_pr_curve(self.model, self.handler)
        min_acc, maj_acc = generate_test_stats(self.model, self.handler, best_threshold)

        if use_wandb:
            wandb.log({"pr_auc": pr_auc, "val_recall": best_recall, "best_precision": best_precision, 'min_acc': min_acc, 'maj_acc': maj_acc})
            wandb.summary['pr_auc'] = pr_auc
            wandb.summary['best_recall'] = best_recall
            wandb.summary['best_precision'] = best_precision
            wandb.summary['min_acc'] = min_acc
            wandb.summary['maj_acc'] = maj_acc

    def save_model(self, name):
        """
        Saves the trained model.
        
        Args:
            name (str): Name of the model file.
        """
        torch.save(self.model, f'{name}.pth')

class SideTuneExperiment(Experiment):
    """
    A class to handle the setup and execution of a side-tuning experiment.
    
    Attributes:
        base_handler (CustomDataHandler): Data handler for base model training data.
        side_handler (CustomDataHandler): Data handler for side model training data.
        config (dict): Configuration parameters for the experiment.
        id_number (str): Identifier for the experiment.
        experiment_type (str): Type of the experiment.
        name (str): Name of the experiment.
        device (torch.device): Device to run the experiment on (CPU or GPU).
    """
    def __init__(self, base_handler, side_handler, config, i):
        """
        Initializes the SideTuneExperiment class with the given parameters.
        
        Args:
            base_handler (CustomDataHandler): Data handler for base model training data.
            side_handler (CustomDataHandler): Data handler for side model training data.
            config (dict): Configuration parameters for the experiment.
            i (int): Identifier for the experiment.
        """
        self.base_handler   = base_handler
        self.side_handler   = side_handler
        self.config         = config
        self.id_number = f"{i}"
        self.experiment_type = f"sidetune"
        self.name = self.experiment_type + "_" + self.id_number
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def generate_experiment(primary_handler, secondary_handler, config, i):
        """
        Generates an instance of the SideTuneExperiment class.
        
        Args:
            primary_handler (CustomDataHandler): Primary data handler.
            secondary_handler (CustomDataHandler): Secondary data handler.
            config (dict): Configuration parameters for the experiment.
            i (int): Identifier for the experiment.
        
        Returns:
            SideTuneExperiment: An instance of the SideTuneExperiment class.
        """
        return SideTuneExperiment(secondary_handler, primary_handler, config, i)

    def run_experiment(self, use_wandb, show_visuals, sweep, use_profiler):
        """
        Runs the side-tuning experiment.
        
        Args:
            use_wandb (bool): Flag to use Weights and Biases for logging.
            show_visuals (bool): Flag to show training visuals.
            sweep (bool): Flag to indicate if the experiment is part of a sweep.
            use_profiler (bool): Flag to use profiler for performance analysis.
        """
        config = self.config

        if use_wandb and not sweep:
            self.start_wandb(self.name)

        if self.config['alpha'] is None:
            config['base_alpha'] = calculate_focal_alpha(self.base_handler)
            config['side_alpha'] = calculate_focal_alpha(self.side_handler)
        else:
            config['base_alpha'] = config['alpha']
            config['side_alpha'] = config['alpha']

        base_model_in, base_model_out = self.get_network_shape(self.base_handler)
        side_model_in, side_model_out = self.get_network_shape(self.side_handler)

        base_model = self.build_model("Flexible", [base_model_in, base_model_out, config['hidden_size'], config['hidden_layers'], config['weight_decay'], config['dropout']])
        side_model = self.build_model("Flexible", [side_model_in, side_model_out, config['hidden_size'], config['hidden_layers'], config['weight_decay'], config['dropout']])
        
        base_model.to(self.device)
        side_model.to(self.device)
        
        side_tune_wrap = SideTuneWrapper(base_model, side_model, alpha = 0.5)

        base_criterion = self.build_criterion(self.config['criterion_type'], [config['base_alpha'], config['gamma']])
        side_criterion = self.build_criterion(self.config['criterion_type'], [config['side_alpha'], config['gamma']])

        base_criterion.to(self.device)
        side_criterion.to(self.device)

        base_optimizer = self.build_optimizer(self.config['optimizer_type'], [base_model.parameters(), config['lr']])
        side_optimizer = self.build_optimizer(self.config['optimizer_type'], [[{'params': side_tune_wrap.side_model.parameters()},
                                                    {'params': side_tune_wrap.side_tuning_network.alpha}],
                                                  config['lr']])

        train_class = Train(side_tune_wrap.base_model, self.base_handler, self.base_handler, 
                            base_optimizer, base_criterion, config['num_epochs'], 
                            use_wandb, show_visuals)
        train_class.train(use_profiler)

        train_class_2 = Train(side_tune_wrap.side_tuning_network, self.side_handler, self.side_handler,
                            side_optimizer, side_criterion, config['num_epochs'],
                            use_wandb, show_visuals)
        train_class_2.train(use_profiler)
        
        self.model = side_tune_wrap.side_tuning_network
        side_tune_wrap.side_tuning_network.to(self.device)
        
        best_precision, best_recall, best_threshold, pr_auc = build_pr_curve(side_tune_wrap.side_tuning_network, self.side_handler)
        min_acc, maj_acc = generate_test_stats(side_tune_wrap.side_tuning_network, self.side_handler, best_threshold)

        if use_wandb:
            wandb.log({"pr_auc": pr_auc, "val_recall": best_recall, "best_precision": best_precision, 'min_acc': min_acc, 'maj_acc': maj_acc})
            wandb.summary['pr_auc'] = pr_auc
            wandb.summary['best_recall'] = best_recall
            wandb.summary['best_precision'] = best_precision
            wandb.summary['min_acc'] = min_acc
            wandb.summary['maj_acc'] = maj_acc
  
class FeatureExtractExperiment(Experiment):
    """
    A class to handle the setup and execution of a feature extraction experiment.
    
    Attributes:
        train_handler (CustomDataHandler): Data handler for training data.
        val_handler (CustomDataHandler): Data handler for validation data.
        id_number (str): Identifier for the experiment.
        experiment_type (str): Type of the experiment.
        name (str): Name of the experiment.
        config (dict): Configuration parameters for the experiment.
        device (torch.device): Device to run the experiment on (CPU or GPU).
    """
    def __init__(self, train_handler, val_handler, config, i):
        """
        Initializes the FeatureExtractExperiment class with the given parameters.
        
        Args:
            train_handler (CustomDataHandler): Data handler for training data.
            val_handler (CustomDataHandler): Data handler for validation data.
            config (dict): Configuration parameters for the experiment.
            i (int): Identifier for the experiment.
        """
        self.train_handler  = train_handler
        self.val_handler    = val_handler
        self.id_number = f"{i}"
        self.experiment_type = f"featureextract"
        self.name = self.experiment_type + "_" + self.id_number
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def generate_experiment(primary_handler, secondary_handler, params, i):
        """
        Generates an instance of the FeatureExtractExperiment class.
        
        Args:
            primary_handler (CustomDataHandler): Primary data handler.
            secondary_handler (CustomDataHandler): Secondary data handler.
            params (dict): Configuration parameters for the experiment.
            i (int): Identifier for the experiment.
        
        Returns:
            FeatureExtractExperiment: An instance of the FeatureExtractExperiment class.
        """
        return FeatureExtractExperiment(secondary_handler, primary_handler, params, i)

    def run_experiment(self, use_wandb, show_visuals, sweep, use_profiler):
        """
        Runs the feature extraction experiment.
        
        Args:
            use_wandb (bool): Flag to use Weights and Biases for logging.
            show_visuals (bool): Flag to show training visuals.
            sweep (bool): Flag to indicate if the experiment is part of a sweep.
            use_profiler (bool): Flag to use profiler for performance analysis.
        """
        config = self.config

        if use_wandb and not sweep:
            self.start_wandb(self.name)

        if config['alpha'] is None:
            config['alpha'] = calculate_focal_alpha(self.train_handler)      

        model_in, model_out = self.get_network_shape(self.train_handler)

        self.model       = self.build_model(config['model_type'], [model_in, model_out, config['hidden_size'], config['weight_decay'], config['dropout']])
        self.model.to(self.device)

        criterion   = self.build_criterion(config['criterion_type'], [config['alpha'], config['gamma']])
        criterion.to(self.device)

        optimizer   = self.build_optimizer(config['optimizer_type'], [self.model.parameters(), config['lr']])

        train_class = Train(self.model, self.train_handler, self.val_handler, optimizer, criterion, config['num_epochs'], use_wandb, show_visuals)
        train_class.train(use_profiler)

        print("replacing the last layer of the model")
        print(self.model)
        for param in self.model.parameters():
            param.requires_grad = False
        print(self.model)

        self.model.fc2 = torch.nn.Linear(self.model.hidden_size, 1)
        for param in self.model.fc2.parameters():
            param.requires_grad = True
        
        print(self.model)

        print("last layers are now changable again")
        self.model.to(self.device)

        optimizer   = self.build_optimizer(config['optimizer_type'], [self.model.parameters(), config['lr']])
        print("optimizer is now created")

        print(self.model)

        train_class = Train(self.model, self.val_handler, self.val_handler, optimizer, criterion, 50, use_wandb, show_visuals)
        train_class.train(use_profiler)

        best_precision, best_recall, best_threshold, pr_auc = build_pr_curve(self.model, self.val_handler)
        min_acc, maj_acc = generate_test_stats(self.model, self.val_handler, best_threshold)

        if use_wandb:
            wandb.log({"pr_auc": pr_auc, "val_recall": best_recall, "best_precision": best_precision, 'min_acc': min_acc, 'maj_acc': maj_acc})
            wandb.summary['pr_auc'] = pr_auc
            wandb.summary['best_recall'] = best_recall
            wandb.summary['best_precision'] = best_precision
            wandb.summary['min_acc'] = min_acc
            wandb.summary['maj_acc'] = maj_acc

class FineTuneExperiment(Experiment):
    """
    A class to handle the setup and execution of a fine-tuning experiment.
    
    Attributes:
        base_handler (CustomDataHandler): Data handler for base model training data.
        fine_handler (CustomDataHandler): Data handler for fine-tuning data.
        id_number (str): Identifier for the experiment.
        experiment_type (str): Type of the experiment.
        name (str): Name of the experiment.
        config (dict): Configuration parameters for the experiment.
        device (torch.device): Device to run the experiment on (CPU or GPU).
    """
    def __init__(self, base_handler, fine_handler, config, i):
        """
        Initializes the FineTuneExperiment class with the given parameters.
        
        Args:
            base_handler (CustomDataHandler): Data handler for base model training data.
            fine_handler (CustomDataHandler): Data handler for fine-tuning data.
            config (dict): Configuration parameters for the experiment.
            i (int): Identifier for the experiment.
        """
        self.base_handler = base_handler
        self.fine_handler = fine_handler
        self.id_number = f"{i}"
        self.experiment_type = f"finetune"
        self.name = self.experiment_type + "_" + self.id_number
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def generate_experiment(primary_handler, secondary_handler, config, i):
        """
        Generates an instance of the FineTuneExperiment class.
        
        Args:
            primary_handler (CustomDataHandler): Primary data handler.
            secondary_handler (CustomDataHandler): Secondary data handler.
            config (dict): Configuration parameters for the experiment.
            i (int): Identifier for the experiment.
        
        Returns:
            FineTuneExperiment: An instance of the FineTuneExperiment class.
        """
        return FineTuneExperiment(secondary_handler, primary_handler, config, i)

    def run_experiment(self, use_wandb, show_visuals, sweep, use_profiler):
        """
        Runs the fine-tuning experiment.
        
        Args:
            use_wandb (bool): Flag to use Weights and Biases for logging.
            show_visuals (bool): Flag to show training visuals.
            sweep (bool): Flag to indicate if the experiment is part of a sweep.
            use_profiler (bool): Flag to use profiler for performance analysis.
        """
        config = self.config
        
        if use_wandb and not sweep:
            self.start_wandb(self.name)

        if self.config['alpha'] is None:
            config['base_alpha'] = calculate_focal_alpha(self.base_handler)
            config['fine_alpha'] = calculate_focal_alpha(self.fine_handler)
        else:
            config['base_alpha'] = config['alpha']
            config['fine_alpha'] = config['alpha']

        model_in, model_out = self.get_network_shape(self.base_handler)
        
        self.model = self.build_model("Flexible", [model_in, model_out, config['hidden_size'], config['hidden_layers'],  config['weight_decay'], config['dropout']])
        self.model.to(self.device)

        base_criterion   = self.build_criterion(config['criterion_type'], [config['base_alpha'], config['gamma']])
        fine_criterion   = self.build_criterion(config['criterion_type'], [config['fine_alpha'], config['gamma']])

        base_criterion.to(self.device)
        fine_criterion.to(self.device)

        base_optimizer = self.build_optimizer(config['optimizer_type'], [self.model.parameters(), config['lr']])
        fine_optimizer = self.build_optimizer(config['optimizer_type'], [self.model.parameters(), config['lr']])

        train_class = Train(self.model, self.base_handler, self.base_handler, base_optimizer, base_criterion, config['num_epochs'], use_wandb, show_visuals)
        train_class.train(use_profiler)

        train_class_2 = Train(self.model, self.fine_handler, self.fine_handler, fine_optimizer, fine_criterion, config['num_epochs'], use_wandb, show_visuals)
        train_class_2.train(use_profiler)

        best_precision, best_recall, best_threshold, pr_auc = build_pr_curve(self.model, self.fine_handler)
        min_acc, maj_acc = generate_test_stats(self.model, self.fine_handler, best_threshold)

        if use_wandb:
            wandb.log({"pr_auc": pr_auc, "val_recall": best_recall, "best_precision": best_precision, 'min_acc': min_acc, 'maj_acc': maj_acc})
            wandb.summary['pr_auc'] = pr_auc
            wandb.summary['best_recall'] = best_recall
            wandb.summary['best_precision'] = best_precision
            wandb.summary['min_acc'] = min_acc
            wandb.summary['maj_acc'] = maj_acc

def generate_experiments(experiment_types, handler1, handler2, config, duplicates):
    """
    Generates a list of experiments.
    
    Args:
        experiment_types (list): List of experiment classes.
        handler1 (CustomDataHandler): Primary data handler.
        handler2 (CustomDataHandler): Secondary data handler.
        config (dict): Configuration parameters for the experiments.
        duplicates (int): Number of duplicates for each experiment type.
    
    Returns:
        list: List of experiment instances.
    """
    experiments = []
    for experiment_type in experiment_types:
        for i in range(duplicates):
            experiments += [experiment_type.generate_experiment(handler1, handler2, config, i+1)]
    return experiments

class Sweep:
    """
    A class to handle the setup and execution of a hyperparameter sweep.
    
    Attributes:
        primary_handler (CustomDataHandler): Primary data handler.
        secondary_handler (CustomDataHandler): Secondary data handler.
        sweep_config (dict): Configuration parameters for the sweep.
        sweep_parameters (dict): Parameters to sweep over.
        metric (dict): Metric to optimize during the sweep.
        project (str): Name of the project in Weights and Biases.
    """
    def __init__(self, primary_handler, secondary_handler, sweep_config, sweep_parameters, metric, project):
        """
        Initializes the Sweep class with the given parameters.
        
        Args:
            primary_handler (CustomDataHandler): Primary data handler.
            secondary_handler (CustomDataHandler): Secondary data handler.
            sweep_config (dict): Configuration parameters for the sweep.
            sweep_parameters (dict): Parameters to sweep over.
            metric (dict): Metric to optimize during the sweep.
            project (str): Name of the project in Weights and Biases.
        """
        self.i = 0
        self.primary_handler    = primary_handler
        self.secondary_handler  = secondary_handler
        sweep_config['metric'] = metric
        sweep_config['parameters'] = sweep_parameters
        self.sweep_config = sweep_config
        self.project = project

    def run(self):
        """
        Runs the hyperparameter sweep.
        """
        sweep_id = wandb.sweep(self.sweep_config, project= self.project)
        wandb.agent(sweep_id, self.perform_sweep, count=500)

    def perform_sweep(self, config=None):
        """
        Performs a single run of the sweep.
        
        Args:
            config (dict, optional): Configuration parameters for the run. Defaults to None.
        """
        wandb.init(config=config,
                resume=True)
        config = wandb.config

        wandb.run.name = config['experiment_type']
        wandb.run.name = config['experiment_type'] + "_" + str(config['repetitions'])

        match config['experiment_type']:
            case 'regular':
                experiment_type  = Experiment
            case 'featureextract':
                experiment_type  = FeatureExtractExperiment
            case 'finetune':
                experiment_type  = FineTuneExperiment
            case 'sidetune':
                experiment_type  = SideTuneExperiment

        config['alpha'] = None
        experiment = experiment_type.generate_experiment(self.primary_handler, self.secondary_handler, config, self.i)
        experiment.run_experiment(use_wandb=True, show_visuals=False, sweep=True, use_profiler=False)
        experiment.save_model("saved_models/"+experiment.name)
        self.i += 1


