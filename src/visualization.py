import matplotlib.pyplot as plt

class Visualize:

    # Scatter diagram suggesting predicted values against actual values of the dataset
    def plot_scatter(self, actual, predicted):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, color='blue', label='Data Points')
        plt.plot(actual, actual, color='red', linewidth=2, label='Perfect Fit')

        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()

