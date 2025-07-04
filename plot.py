import matplotlib.pyplot as plt
from IPython.display import clear_output

class LossPlotter:
    def __init__(self):
        self.trainLosses = []
        self.valLosses = []
        self.epochs = []

    def add(self, epoch, trainLoss, valLoss=None):
        self.epochs.append(epoch)
        self.trainLosses.append(trainLoss)

    def plot(self, valLoss=None):
        if valLoss is not None:
            self.valLosses.append(valLoss)

        plt.figure(figsize=(10, 5))

        plt.plot(self.epochs, self.trainLosses, label='Train Loss', color='blue')
        if valLoss is not None:
            plt.plot(self.epochs, self.valLosses, label='Val Loss', color='orange')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.grid(True)
        plt.savefig('logs/lastLoss.png')
        plt.close()
