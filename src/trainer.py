import torch.nn as nn
import torch
import os
import json
import re
import numpy as np

class Trainer:
    def __init__(self, num_training_iterations, learning_rate, batch_size, model, training_data_path, device="cpu"):
        # self._model = model.to(device) # TODO
        
        self._num_training_iterations = num_training_iterations
        self._model = model
        self._optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self._device = device
        self._training_batches = np.load(os.path.join(training_data_path, "data.npy"))
        print(self._training_batches.shape)
        self._anomalies = np.load(os.path.join(training_data_path, "anomalies.npy"))

    def run_training(self):
        """
        Run the training loop for a specified number of batches.
        """
        print("Training")
        for iteration in range(self._num_training_iterations):
            batch = self._training_batches[iteration][0]
            batch = torch.from_numpy(batch).float().unsqueeze(0).to(self._device)
            loss = self._train_step(batch)
            print(f"Iteration {iteration}, Loss: {loss.item()}")

        self.save_experiment(self._model, {
            "num_training_iterations": self._num_training_iterations,
            "learning_rate": self._optimizer.param_groups[0]['lr'],
            "batch_size": self._training_batches.shape[0]
        }, loss)

        return loss

    def save_experiment(self, model, hyperparameters, loss):
        """
        Create a directory for the experiment and save model, hyperparameters, and loss.
        """
        # Create experiment directory, lookup appropriate number
        checkpoint_dir = "experiments"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        existing_experiments = [d for d in os.listdir(checkpoint_dir) if re.match(r'experiment_\\d+', d)]
        if existing_experiments:
            nums = [int(re.findall(r'\\d+', name)[0]) for name in existing_experiments]
            next_experiment_num = max(nums) + 1
        else:
            next_experiment_num = 1
        exp_dir = os.path.join(checkpoint_dir, f"experiment_{next_experiment_num}")
        os.makedirs(exp_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(exp_dir, "model.pt")
        torch.save(model.state_dict(), model_path)

        # Save hyperparameters
        hparams_path = os.path.join(exp_dir, "hyperparameters.json")
        with open(hparams_path, 'w') as f:
            json.dump(hyperparameters, f, indent=2)

        # Save loss (loss)
        loss_path = os.path.join(exp_dir, "loss.txt")
        with open(loss_path, 'w') as f:
            f.write(str(loss.item()))

        print(f"Experiment saved to {exp_dir}")

    def _compute_loss(self, prediction, target):
        """
        Compute the mean squared error loss between prediction and target.

        NOTE: We're training a model to reconstruct the original input -- the idea is that anomalies will have a higher reconstruction error.
        """
        # Computes a single value of loss
        return nn.MSELoss()(prediction, target)

    def _train_step(self, training_batch):
        # Set the model's mode to train
        self._model.train()

        # Zero gradients for every step
        self._optimizer.zero_grad()

        # Forward pass
        prediction = self._model(training_batch)

        # Compute the loss
        loss = self._compute_loss(prediction, training_batch)

        # Backward pass
        '''TODO: complete the gradient computation and update step.
            Remember that in PyTorch there are two steps to the training loop:
            1. Backpropagate the loss
            2. Update the model parameters using the optimizer
        '''
        # This computes the gradient of the loss with respect to model parameters
        loss.backward()
        # This updates model parameters using the optimizer
        self._optimizer.step()

        return loss