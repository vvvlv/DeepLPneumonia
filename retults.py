import dataset
import network
import loading_augmentation

training_accuracy = np.array([0.5305, 0.5897, 0.7759, 0.8741, 0.8845, 0.9023, 0.8937, 0.9224,
 0.919 , 0.927 , 0.9351, 0.9374, 0.9402, 0.942 , 0.9529]) # Our most successful epoch was number 12

validation_accuracy = np.array([0.5   , 0.9091, 0.9182, 0.9273, 0.9409, 0.75  , 0.9409, 0.9409,
 0.95  , 0.9273, 0.9364, 0.9545, 0.9545, 0.9273, 0.9545])

epochs = range(1, len(training_accuracy) + 1) # Making a range from 1 to the length of our array to get number of epochs

plt.plot(epochs, training_accuracy, 'b', label=f'Training Accuracy')
plt.plot(epochs, validation_accuracy, 'g', label=f'Validation Accuracy')
plt.title(f'Training and Validation Accuracy' )
plt.xlabel(f'Epochs')
plt.ylabel(f'Accuracy')
plt.legend()

plt.show()


Training_data = np.array([1, 2, 3]) ## TODO: INPUT REAL NUMBERS
Validation_data = np.array([1, 2, 3]) ## TODO: INPUT REAL NUMBERS

def plot_metric(metric_name, training_array, validation_array):
    """
    Plots the given metric for both training and validation data using Matplotlib.
    
    Parameters:
    - metric_name: The name of the metric to plot (e.g., 'loss' or 'accuracy').
    - training_array: array of the loss/accuracy values for each epoch collected from training set.
    - validation_array: array of the loss/accuracy values for each epoch collected from validation set.
    """

                    
    num_epochs = len(training_array) #getting the amount of epoches from an input array
    epochs = range(1, num_epochs + 1) #sequence of epoch numbers for the plot
    plt.plot(epochs, training_array, 'b', label=f'Training {metric_name}')
    plt.plot(epochs, validation_array, 'g', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}' )
    plt.xlabel('Epochs')
    plt.ylabel(f' {metric_name}')
    plt.legend()

    plt.show()


    ## We need true/false positives/negatives

actual_labels = [1, 0, 1, 0, 1, 1, 1, 1] # TODO: make this real vector
predicted_labels = [0, 1, 1, 0, 1, 1, 1, 1] # TODO: make this real vector


def show_confusion_matrix(actual_labels, predicted_labels):
    """
    This function prints the confusion matrix.
    :param actual_labels: array of 0(label 'Pneumonia') and 1(label 'Healthy') given with dataset.
    :param predicted_labels: array of 0(label 'Pneumonia') and 1(label 'Healthy') predicted by model.
    """
    cm = confusion_matrix(actual_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Healthy', 'Pneumonia'], 
            yticklabels=['Healthy', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
show_confusion_matrix(actual_labels, predicted_labels)