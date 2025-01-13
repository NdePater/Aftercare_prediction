import numpy as np
config = {
  'hidden_size': 64,
  'hidden_layers': 1,
  'weight_decay': 0.001,
  'dropout': 0.0,
  'fc_layer_size' : 64,
  'lr' : 1e-4,
  'alpha' : None,
  'gamma' : 2,
  'num_epochs' : 100,
  'model_type' : 'NeuralNetwork',
  'optimizer_type' : 'Adam',
  'criterion_type' : 'FocalLoss',
  'apply_smote': False
}

# regular sweep_parameters
sweep_parameters = {
  'hidden_size': {
    #   'values': [64]  # 256
    'values': [64, 128, 256]  # 256

  },
  'weight_decay': {
      'values': [0.001]
  },
  'dropout': {
      'values': [0.0]
  },
  'lr' : {
      'values':[1e-3]
  },
#   'alpha' : {
#       'value': None
#   },
  'gamma' : {
      'values': [1, 2]
  },
  'num_epochs' : {
      'values': [300]
  },
  'model_type' : {
      'values': ['NeuralNetwork']
  },
  'optimizer_type' : {
      'values': ['Adam']
  },
  'criterion_type' : {
      'values': ['FocalLoss']
  },
  'experiment_type' : {
      'values' : ['regular', 'sidetune', 'featureextract', 'finetune']
  },
  'repetitions' :{
      'values' : [i for i in range(1)]
  },
  'apply_smote' : {
    'values' : [False]
  }
}

parameters_bayesian = {
  'hidden_size': {
    'min': 32,
    'max': 512,
  },
  'hidden_layers' : {
      'values': [1, 2, 3]
  },
  'weight_decay': {
    'min': 0.0,
    'max': 0.5,
    'distribution': 'uniform'
  },
  'dropout': {
    'min': 0.0,
    'max': 0.5,
  },
  'lr' : {
    'min': 1e-5,
    'max': 1e-2,
    # 'distribution': 'log_uniform' 
  },
  'gamma' : {
      'values': [1, 2]
  },
  'num_epochs' : {
      'values': [300]
  },
  'model_type' : {
      'values': ['NeuralNetwork']
  },
  'optimizer_type' : {
      'values': ['Adam']
  },
  'criterion_type' : {
      'values': ['FocalLoss']
  },
  'experiment_type' : {
      'values' : ['regular', 'sidetune', 'featureextract', 'finetune']
  },
  'repetitions' :{
      'values' : [i for i in range(1)]
  },
  'apply_smote' : {
    'values' : [False]
  }
}

num_epochs = 1000
batch_size = 2048
hidden_size = 128
alpha = 0.55
gamma = 2
weight_decay = 0.0001
val_type = "split"
val_size = 0.2
test_year = 2023
val_year = 2022
start_date= 2013
apply_scaler = True
apply_smote = False
lr = 1e-6
drop = 0.5
