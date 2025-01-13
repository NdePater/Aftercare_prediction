import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class CustomDataset(Dataset):
    """
    Custom Dataset class for PyTorch.

    Attributes:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
    """
    def __init__(self, X, y):
        """
        Initialize the dataset with features and target.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.
        """
        self.X = X
        self.y = y

    def __getitem__(self, index):
        """
        Get item by index.

        Args:
            index (int): Index.

        Returns:
            tuple: (feature, target) at the given index.
        """
        return self.X[index], self.y[index]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.X)

class CustomDataHandler():
    """
    Custom Data Handler class for managing dataset operations.

    Attributes:
        data (pd.DataFrame): The entire dataset.
        features (list): List of feature column names.
        target (str): Target column name.
        test_size (float): Proportion of the dataset to include in the test split.
        batch_size (int): Number of samples per batch.
        frac_train (float): Fraction of training data to use.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data, features, target, test_size, batch_size, frac_train, transform=None):
        """
        Initialize the data handler with dataset and parameters.

        Args:
            data (pd.DataFrame): The entire dataset.
            features (list): List of feature column names.
            target (str): Target column name.
            test_size (float): Proportion of the dataset to include in the test split.
            batch_size (int): Number of samples per batch.
            frac_train (float): Fraction of training data to use.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.features = features
        self.target = target
        self.test_size = test_size
        self.batch_size = batch_size
        self.frac_train = frac_train

    def split_data(self, test_year, val_type="split", val_year=None, select_features=None):
        """
        Split the data into training, validation, and test sets.

        Args:
            test_year (int): Year to use for the test set.
            val_type (str, optional): Type of validation split ("split" or "year"). Defaults to "split".
            val_year (int, optional): Year to use for the validation set if val_type is "year". Defaults to None.
            select_features (list, optional): List of selected features. Defaults to None.
        """
        if select_features is not None:
            med_cols = self.data.filter(regex=r'^MED_').columns
            icd_cols = self.data.filter(regex=r'^ICD_').columns
            dbc_cols = self.data.filter(regex=r'^DBC_').columns

            self.features = select_features + list(med_cols) + list(icd_cols) + list(dbc_cols)

        test_year_mask = self.data['Opname jaar'] == test_year
        val_year_mask = self.data['Opname jaar'] == val_year

        X = self.data[self.features]
        y = self.data[self.target]

        id = self.data["Patiëntnummer"]

        if val_type == "split":
            X_train, X_val, y_train, y_val = train_test_split(X[~test_year_mask], y[~test_year_mask], test_size=self.test_size, random_state=1)
        elif val_type == "year":
            X_train = X[~test_year_mask & ~val_year_mask]
            y_train = y[~test_year_mask & ~val_year_mask]
            X_val = X[val_year_mask]
            y_val = y[val_year_mask]

        train_sub_size = int(len(X_train) * self.frac_train)
        train_indices = random.sample(range(len(X_train)), train_sub_size)
        if self.frac_train != 1:
            print(f"sampled {len(train_indices)} items from total {len(X_train)} which is {self.frac_train} of data")

        X_train = X_train.iloc[train_indices]
        y_train = y_train.iloc[train_indices]

        X_test = X[test_year_mask]
        y_test = y[test_year_mask]
        self.id_test = id[test_year_mask]

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = X_train, y_train, X_val, y_val, X_test, y_test

        self.y_val_array = self.y_val.values
        self.y_val_min_mask = self.y_val_array == 1
        self.y_val_min = self.y_val_array[self.y_val_min_mask]
        self.y_val_maj = self.y_val_array[~self.y_val_min_mask]

    def apply_scaler(self):
        """
        Apply standard scaling to the features.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)

        self.X_train, self.X_val, self.X_test = [self.scaler.transform(i) for i in [self.X_train, self.X_val, self.X_test]]

    def get_inverse_scaler(self, x):
        """
        Inverse transform the scaled data.

        Args:
            x (np.ndarray): Scaled data.

        Returns:
            np.ndarray: Original data.
        """
        return self.scaler.inverse_transform(x)

    def make_tensors(self):
        """
        Convert the data to PyTorch tensors.
        """
        self.X_train_tensor, self.X_val_tensor, self.X_test_tensor = [torch.Tensor(i) for i in [self.X_train, self.X_val, self.X_test]]
        self.y_train_tensor, self.y_val_tensor, self.y_test_tensor = [torch.LongTensor(i.values) for i in [self.y_train, self.y_val, self.y_test]]

    def apply_smote(self):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data.
        """
        print("Applied SMOTE")
        smote = SMOTE(random_state=1)
        self.unsmoted_X_train_tensor = self.X_train_tensor.clone()
        self.unsmoted_y_train_tensor = self.y_train_tensor.clone()
        self.X_train_tensor, self.y_train_tensor = smote.fit_resample(self.X_train_tensor, self.y_train_tensor)
        self.create_custom_datasets()
        self.create_dataloaders()

    def remove_smote(self):
        """
        Remove SMOTE and revert to the original training data.
        """
        if hasattr(self, 'unsmoted_X_train_tensor'):
            print("Successfully removed SMOTE")
            self.X_train_tensor = self.unsmoted_X_train_tensor.clone()
            self.y_train_tensor = self.unsmoted_y_train_tensor.clone()
            self.create_custom_datasets()
            self.create_dataloaders()
        else:
            print("No SMOTE applied")

    def create_custom_datasets(self):
        """
        Create custom datasets for training, validation, and test sets.
        """
        self.train_dataset = CustomDataset(self.X_train_tensor, self.y_train_tensor)
        self.val_dataset = CustomDataset(self.X_val_tensor, self.y_val_tensor)
        self.test_dataset = CustomDataset(self.X_test_tensor, self.y_test_tensor)

    def create_dataloaders(self):
        """
        Create data loaders for training, validation, and test sets.
        """
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

def load_data(file, start_date, val_size, batch_size, frac_train):
    """
    Load data from a file and prepare it for training.

    Args:
        file (str): Path to the data file.
        start_date (int): Start year for filtering the data.
        val_size (float): Proportion of the dataset to include in the validation split.
        batch_size (int): Number of samples per batch.
        frac_train (float): Fraction of training data to use.

    Returns:
        tuple: Two CustomDataHandler objects for different groups.
    """
    if file[-3:] == 'csv':
        df_prep = pd.read_csv(file, delimiter=',')
    elif file[-4:] == 'xlsx':
        df_prep = pd.read_excel(file)
    else:
        print(f'unknown extension: {file[-4:]}')

    non_feature_columns = ["Patiëntnummer", "OpnameDatumTijd", "Opnamespecialisme"]
    non_integer_columns = ['DBCcode', 'ICD_CODE']

    features = [a for a in df_prep.columns if a not in non_feature_columns + non_integer_columns]

    # convert all features columns to float
    df_prep[features].astype(float)

    groups = "Opnamespecialisme"
    target = 'Ontslagbestemming'

    df_prep['Opname jaar'] = pd.to_datetime(df_prep['OpnameDatumTijd']).dt.year
    data = df_prep[df_prep['Opname jaar'] > start_date]

    df_chi = data[data[groups] == 'CHI']
    df_ort = data[data[groups] == 'ORT']

    chi_handler = CustomDataHandler(df_chi, features, target, val_size, batch_size, frac_train)
    ort_handler = CustomDataHandler(df_ort, features, target, val_size, batch_size, frac_train)

    return chi_handler, ort_handler

def prepare_data(handlers, test_year, val_year, val_type, apply_scaler, apply_smote, select_features=None):
    """
    Prepare data for training by splitting, scaling, and applying SMOTE.

    Args:
        handlers (list): List of CustomDataHandler objects.
        test_year (int): Year to use for the test set.
        val_year (int): Year to use for the validation set.
        val_type (str): Type of validation split ("split" or "year").
        apply_scaler (bool): Whether to apply scaling.
        apply_smote (bool): Whether to apply SMOTE.
        select_features (list, optional): List of selected features. Defaults to None.
    """
    for handler in handlers:
        handler.split_data(test_year, val_type, val_year, select_features)
        if apply_scaler:
            handler.apply_scaler()
        handler.make_tensors()
        if apply_smote:
            handler.apply_smote()
        else:
            handler.create_custom_datasets()
            handler.create_dataloaders()

def get_fulldata_from_loader(loader):
    """
    Get the full dataset from a data loader.

    Args:
        loader (DataLoader): Data loader.

    Returns:
        tuple: Full features and target tensors.
    """
    X, y = [], []
    for X_batch, y_batch in loader:
        X.append(X_batch)
        y.append(y_batch)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)