"""
Contains functions for training a PyTorch model.
"""
import torch
import time
import numpy as np
import utils
#from PytorchModular import utils

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange


class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 dataloaders: torch.utils.data.DataLoader,
                 epochs: int, 
                 criterion: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 writer: torch.utils.tensorboard,
                 save_dir: str,
                 device: torch.device):
        """
        Initial a Trainer class contains functions for training and testing a PyTorch model.

        Args:
            model: A PyTorch model to be trained and tested.
            dataloaders: A dictionary of DataLoaders for the model to be trained on (dataloaders['train']).
            epochs: An integer indicating how many epochs to train for.
            critierion: A PyTorch loss function to calculate loss on both datasets.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            scheduler: A PyTorch learning rate scheduler to help schedule the lr.
            writer: A tensorboard SummaryWriter to store training statistics.
            save_dir: A Str indicating the save directory of the models.
            device: A target device to compute on (e.g. "cuda" or "cpu").

        Example usage:
            Trainer(model=model_ft, dataloaders=dataloaders, epochs=5,
                    criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, 
                    writer=writer, save_dir='/content/OutputModel', device=device)
        """
        
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.curr_epoch = 0
        self.epochs = epochs
        self.writer = writer
        self.save_dir = save_dir
        self.device = device
        # Create empty results dictionary
        self.results = {"train_loss": [],
                        "train_acc": [],
                        "val_loss": [],
                        "val_acc": []
                        }
        
    def train(self):
        """
        Passes a target PyTorch models through train_step() and test_step() functions for a number of epochs.

        Returns:
            A dictionary of training and testing loss as well as training and testing accuracy metrics. 
            Each metric has a value in a list for each epoch.

            For example if training for epochs=2: 
                {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 

        Example usage:
            model_results = trainer.train()
            -->
            Epoch 9: Train Acc: 0.9453125 Train loss: 0.12280523835215718 Val Acc: 0.95 Val loss: 0.1460084779188037
            Epoch 10: Train Acc: 0.95703125 Train loss: 0.11239275464322418 Val Acc: 0.95 Val loss: 0.14841991513967515
            --------------------
            Training complete in 0m 27s 
        """
        start_time = time.time()

        progressbar = trange(self.epochs)
        for _ in progressbar:
            # Epoch counter
            self.curr_epoch += 1
            progressbar.set_description(f'Epoch {self.curr_epoch}')

            # Training block
            self.train_epoch()

            # Validation block
            self.val_epoch()

            #  Update the current best metric and loss
            progressbar.set_postfix(BestTrainAcc = np.max(self.results["train_acc"]), BestTrainLoss=np.min(self.results["train_loss"]), 
                                    BestValAcc=np.max(self.results["val_acc"]), BestValLoss = np.min(self.results["val_loss"]))
            # Display the statistics for current Epoch
            print(f'\n Epoch {self.curr_epoch}: Train Acc: {self.results["train_acc"][-1]} Train loss: {self.results["train_loss"][-1]} Val Acc: {self.results["val_acc"][-1]} Val loss: {self.results["val_loss"][-1]}')

            # Save checkpoints every epoch
            save_model_name = f'Model_Epoch_{self.curr_epoch}.pth'
            utils.save_model(self.model, self.save_dir, save_model_name)

        time_elapsed = time.time() - start_time
        print('-' * 20)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s. Saving model to: {self.save_dir}.')

        return self.results
        
    def train_epoch(self):
        """
        Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then runs through all of the required 
        training steps (forward pass, loss calculation, optimizer step).
        """

        # Training mode
        self.model.train()

        # Setup train loss and train accuracy values
        running_accs, running_losses = [], []

        batch_iter = tqdm(
            enumerate(self.dataloaders['train']),
            total=len(self.dataloaders['train']),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            batch_iter.set_description(f'Train Step {i}')
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            # 1. Forward pass
            outputs = self.model(inputs)

            # 2. Calculate and accumulate the loss
            loss = self.criterion(outputs, targets)
            loss_value = loss.item()
            running_losses.append(loss_value)

            # 3. Zero the parameter gradients
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()

            # 6. Calculate the metric
            y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            acc_value = (y_pred_class == targets).sum().item()/len(outputs)
            running_accs.append(acc_value)

        self.scheduler.step()
        self.results["train_loss"].append(np.mean(running_losses))
        self.results["train_acc"].append(np.mean(running_accs))

        # Write the date to TensorBoard log dir
        self.writer.add_scalar("Loss/train", np.mean(running_losses), self.curr_epoch)
        self.writer.add_scalar("Acc/train", np.mean(running_accs), self.curr_epoch)   

    def val_epoch(self):
        """
        Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to "eval" mode and then performs a forward pass on a testing dataset.
        """
        # Validation mode
        self.model.eval()
        running_accs, running_losses = [], []

        batch_iter = tqdm(
            enumerate(self.dataloaders['val']),
            total=len(self.dataloaders['val']),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            batch_iter.set_description(f'Val Step {i}')
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            with torch.no_grad():
                # Forward pass
                outputs = self.model(inputs)

                # Calculate the loss
                loss = self.criterion(outputs, targets)
                loss_value = loss.item()
                running_losses.append(loss_value)

                # Calculate the metric
                y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                acc_value = (y_pred_class == targets).sum().item()/len(outputs)
                running_accs.append(acc_value)

        self.results["val_loss"].append(np.mean(running_losses))
        self.results["val_acc"].append(np.mean(running_accs))
        
        # Write the date to TensorBoard log dir
        self.writer.add_scalar("Loss/val", np.mean(running_losses), self.curr_epoch)
        self.writer.add_scalar("Acc/val", np.mean(running_accs), self.curr_epoch)

