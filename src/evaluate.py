import matplotlib.pyplot as plt
import numpy as np


class Visualize:

    # Scatter diagram suggesting predicted values against actual values of the dataset
    def plot_scatter(self, actual, predicted, model):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, color='blue', label='Data Points')
        plt.plot(actual, actual, color='red', linewidth=2, label='Perfect Fit')

        plt.title(f"{model} :Predicted vs Actual Values")
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Bar chart diagram suggesting the importance of specific columns in the dataset
    # Shows which columns contribute to the most accurate predictions
    def plot_importance(self, model, names):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(names)), importances[indices])
        plt.xticks(range(len(importances)), [names[i] for i in indices], rotation=90)
        plt.tight_layout
        plt.show()




