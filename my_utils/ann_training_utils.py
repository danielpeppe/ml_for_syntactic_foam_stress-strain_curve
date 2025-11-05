import torch
import torch.optim as optim

def train_ann_model(
    model_class,
    device,
    params,
    hyper_dict,
    tensor_dict,
    verbose=False,
):
    """
    Creates and trains an FNN with the specified hidden_layer_sizes.
    Returns the trained model, training losses and test loss.

    inputs:
        model_class: model class
        device: device
        params: parameters
        hidden_layer_sizes: hidden layer sizes
        tensor_dict: dictionary of tensors
        verbose: verbose (default=False)
    return:
        model: trained model
        losses_train: training losses
        loss_test: test loss
    """

    # Extract tensors
    input_train = tensor_dict['input_train_shuffled']
    output_train = tensor_dict['output_train_shuffled']
    input_test = tensor_dict['input_test']
    output_test = tensor_dict['output_test']

    # Extract params
    activation = params['activation']
    criterion = params['criterion']
    learning_rate = params['learning_rate']
    n_epochs = params['n_epochs']
    scheduler_step_size = params['scheduler_step_size']
    scheduler_gamma = params['scheduler_gamma']
    num_batches = params['num_batches']        
    # Extract hyperparams
    hidden_layer_sizes = hyper_dict['candidate_hidden_configs']
    dropout_rate = hyper_dict['dropout_rate']
    weight_decay = hyper_dict['weight_decay']

    # Create model
    input_size = len(params['input_labels'])
    output_size = len(params['output_labels'])
    model = model_class(input_size, hidden_layer_sizes, output_size, activation, dropout_rate)
    model = model.to(device)  # ensure it uses the same device as the data

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler_step_size and scheduler_gamma is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Training loop
    losses_train = []
    model.train()
    for epoch in range(n_epochs):

        batch_losses = []
        for batch_idx in range(num_batches):
            # Get batch
            input_train_batch = input_train[batch_idx, :, :]
            output_train_batch = output_train[batch_idx, :, :]
        
            # Forward pass
            output_pred = model(input_train_batch)
            loss_batch = criterion(output_pred, output_train_batch)

            # Backward pass
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            
            batch_losses.append(loss_batch.item())
        
        # Calculate loss for this epoch as the average loss of the batches
        loss_train = sum(batch_losses) / num_batches
        losses_train.append(loss_train)
        # Update learning rate
        if scheduler_step_size and scheduler_gamma is not None:
            scheduler.step()
        # Print loss
        if verbose and epoch % 200 == 0:
            print(f"Epoch {epoch} | Train loss (avg of {num_batches} batches): {loss_train:.2e}")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(input_test)
        loss_test = criterion(test_preds, output_test).item()

    return model, losses_train, loss_test, test_preds
