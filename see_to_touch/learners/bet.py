import os
import torch

from pathlib import Path

from .learner import Learner 

class BETLearner(Learner):
    def __init__(
        self,
        bet_model,
        optimizer
    ):
        self.optimizer = optimizer 
        self.bet_model = bet_model 

    def to(self, device):
        self.device = device
        self.bet_model.to(device)

    def train(self):
        self.bet_model.train()

    def eval(self):
        self.bet_model.eval()

    def save(self, checkpoint_dir, model_type='best'):
        self.bet_model.save_model(Path(checkpoint_dir)) # This is the requirement for the bet model

    def train_epoch(self, train_loader):
        self.train() 

        # Save the train loss
        train_loss = 0.0 

        # Training loop 
        for batch in train_loader: 
            self.optimizer.zero_grad()
            obs, act = (x.to(self.device) for x in batch)
            # print('obs.shape: {}, act.shape: {}'.format(
            #     obs.shape, act.shape
            # ))

            # Get the loss
            _, loss, loss_dict = self.bet_model(obs, None, act) # It's unconditional
            train_loss += loss.item()
            # print(f'loss_dict: {loss_dict}')

            # Backprop
            loss.backward() 
            self.optimizer.step()

        return train_loss / len(train_loader), loss_dict # Return the last loss dict
    
    def test_epoch(self, test_loader):
        self.eval() 

        # Save the test loss
        test_loss = 0.0 

        # Training loop 
        with torch.no_grad():
            for batch in test_loader: 
                obs, act = (x.to(self.device) for x in batch)

                # Get the loss
                _, loss, loss_dict = self.bet_model(obs, None, act) # It's unconditional
                test_loss += loss.item()

        return test_loss / len(test_loader), loss_dict
