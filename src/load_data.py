import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Function for loading and handling the datasets
class DataLoader:
    def __init__(self, filepath, target):
        self.target = target
        self.filepath = filepath
        self.data = None
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None

    def load_data(self):
        self.data = pd.read_csv(self.filepath, encoding='ISO-8859-1')

    def preprocess_data(self):
        # Removes unnecessary features
        self.data.drop(columns=['Unnamed: 0', 'id', 'address'], inplace=True)
        # Although this is unnecessary it is there in-case, or if the dataset updates
        self.data = self.data.dropna()
        # Encodes categories for more easier interpretation by the computer
        self.data = pd.get_dummies(self.data)

        # Skews any values logarithmically to remove outliers
        for column in self.data.select_dtypes(include=[np.number]).columns:
            if abs(self.data[column].skew()) > 0.5:
                self.data[column] = np.log1p(self.data[column])

    def split_data(self):
        Y = self.data[self.target]
        X = self.data.drop(columns=['price'], axis=1)

        X = (X - X.mean()) / (X.std())

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



    def get_data(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test
