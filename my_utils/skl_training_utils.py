import torch
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

def train_scikit_learn_model(model_class, hyper_dict, tensor_dict, params):
    # Extract tensors
    input_train_shuffled = tensor_dict['input_train_shuffled']
    output_train_shuffled = tensor_dict['output_train_shuffled']
    input_test = tensor_dict['input_test']
    output_test = tensor_dict['output_test']
    # Extract params
    criterion = params['criterion']

    # Create a fresh model with these params
    model_curr = model_class(**hyper_dict)
    # Wrap in MultiOutputRegressor if necessary
    if isinstance(model_curr, SVR):
        if len(output_train_shuffled.shape) != 1:
            model_curr = MultiOutputRegressor(model_curr)
    # Train on the entire training set
    model_curr.fit(input_train_shuffled, output_train_shuffled)
    # Evaluate on your test set
    output_pred_test = model_curr.predict(input_test)
    # Get train loss
    output_pred_train = model_curr.predict(input_train_shuffled)
    loss_train = criterion(torch.as_tensor(output_train_shuffled), torch.as_tensor(output_pred_train)).item()
    # Get test loss
    loss_test = criterion(torch.as_tensor(output_test), torch.as_tensor(output_pred_test)).item()

    return model_curr, loss_train, loss_test