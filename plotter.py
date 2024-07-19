import os
import matplotlib.pyplot as plt

class plotter:
    def __init__(self, directory="plots"):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def AddValue(self, train_loss, train_acc, val_loss, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        print("\n Epoch: ", epoch, " Training Acc: ", train_acc, " | Validation Acc: ", val_acc)

        #--------------------Single Line Print----------------------
        #print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        print(f" Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%")
        #print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        #print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")


    def ShowFig(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 6))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.xticks(epochs)
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Training Accuracy')
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xticks(epochs)
        plt.legend()
        
        # Save the plot to the directory
        #plt.savefig(os.path.join(self.directory, 'training_history.png'))
        #plt.show()
