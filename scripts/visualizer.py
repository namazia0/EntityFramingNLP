import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    """
    Create visualizations for the data.
    """
    def __init__(self, data):
        self.data = data

    def plot(self):
        # Plotting logic
        pass

    def save(self):
        # Saving logic
        pass

    def show(self):
        # Showing logic
        pass

    def run(self):
        self.plot()
        self.save()
        self.show()

def boxplot(data):
    """
    Create a boxplot for list of data.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data)
    plt.title("Performance Metrics Boxplot")
    plt.ylabel("Scores")
    plt.xticks(rotation=20)
    plt.show()