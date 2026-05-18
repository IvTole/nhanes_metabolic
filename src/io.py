import pandas as pd
from typing import Optional, Tuple

# External modules
from src.config import train_data_path
from src.config import FEATURES, TARGET

class Dataset:

    def __init__(self, num_samples: int = None, random_seed: int = 42):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param random_seed: the random seed to use when sampling data points
        """

        self.num_samples = num_samples
        self.random_seed = random_seed

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return: pd.DataFrame (df_train)
        """

        train_path = train_data_path()

        df_train = pd.read_csv(train_path)

        # Sample
        if self.num_samples is not None:
            df_train = df_train.sample(self.num_samples, random_state=self.random_seed)   

        return df_train
    
    def load_data_xy(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        :return: Tuple (X_train, y_train)
        """

        df_train = self.load_data()

        X_train = df_train.drop(columns=[TARGET], axis=1)
        y_train = df_train[TARGET]

        return X_train, y_train