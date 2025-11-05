import torch.nn as nn
import torch
# Model classes
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class FNN(nn.Module):
    '''
    Feedforward Neural Network class.
    '''
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation, dropout_rate):
        super(FNN, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_rate))
        # Intermediate layers
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
        # Output layer
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



def get_model_class_and_device(architecture):
    '''
    Function to get the model class and device.
    '''
    # Default device
    device = 'cpu' # RF doesn't use GPU
    model_class = None

    if architecture == 'ANN':
        # Get CUDA device if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("GPU available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Device name:", torch.cuda.get_device_name())

        # Define FNN class
        print('Defining FNN class')
        model_class = FNN
    elif architecture == 'RF':
        # Define RandomForestRegressor class
        print('Defining RandomForestRegressor class')
        model_class = RandomForestRegressor
    elif architecture == 'SVR':
        # Define SVR class
        print('Defining SVR class')
        model_class = SVR

    return model_class, device