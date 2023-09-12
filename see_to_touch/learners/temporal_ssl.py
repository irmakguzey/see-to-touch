# Learner to implement temporal infonce loss between
# different frames of the trajectories

import os
import torch

from see_to_touch.utils import mse, l1

from info_nce import InfoNCE

from .learner import Learner

# TODO: You can turn these where you're moving things around from your __dict__ or smth
class TemporalSSLLearner(Learner):
    def __init__(
        self,
        optimizer,
        repr_loss_fn, # infonce 
        joint_diff_loss_fn, # mse
        encoder, # Get the image representations 
        linear_layer, # Predict the joint difference given the image reprs 
        joint_diff_scale_factor, # Will be used to scale the joint difference loss with the representation loss
        total_loss_type, # contrastive, joint, contrastive_joint 
    ):
        self.optimizer = optimizer
        self.encoder = encoder
        self.linear_layer = linear_layer
        self.total_loss_type = total_loss_type

        if repr_loss_fn == 'infonce':
            self.repr_loss_fn = InfoNCE()

        if joint_diff_loss_fn == 'mse': 
            self.joint_diff_loss_fn = mse
        elif joint_diff_loss_fn == 'l1':
            self.joint_diff_loss_fn = l1

        self.joint_diff_scale_factor = joint_diff_scale_factor

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.linear_layer.to(device)

    def train(self): 
        self.encoder.train() 
        self.linear_layer.train() 

    def eval(self): 
        self.encoder.eval() 
        self.linear_layer.eval() 

    def save(self, checkpoint_dir, model_type='best'):
        torch.save(self.encoder.state_dict(),
                   os.path.join(checkpoint_dir, f'image_encoder_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)
        
        torch.save(self.linear_layer.state_dict(),
                   os.path.join(checkpoint_dir, f'linear_layer_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)
        
    def train_epoch(self, train_loader):
        self.train() 

        train_loss = 0.

        for batch in train_loader:
            self.optimizer.zero_grad() 
            curr_img, next_img, joint_diff = [b.to(self.device) for b in batch]
            curr_repr = self.encoder(curr_img)
            next_repr = self.encoder(next_img)

            loss = 0
            if 'contrastive' in self.total_loss_type:
                repr_loss = self.repr_loss_fn(
                    curr_repr,
                    next_repr
                )
                loss += repr_loss

            if 'joint' in self.total_loss_type:
                all_repr = torch.concat([curr_repr, next_repr], dim=-1)
                pred_joint_diff = self.linear_layer(all_repr)
                joint_diff_loss = self.joint_diff_loss_fn(joint_diff, pred_joint_diff)
                loss += joint_diff_loss * self.joint_diff_scale_factor

            # print('repr_loss: {}, joint_diff_loss: {}'.format(
            #     repr_loss, joint_diff_loss
            # ))

            # loss = repr_loss + joint_diff_loss * self.joint_diff_scale_factor
            train_loss += loss.item()

            loss.backward() 
            self.optimizer.step()

        return train_loss / len(train_loader)

    def test_epoch(self, test_loader):
        self.eval() 

        test_loss = 0.

        for batch in test_loader:
            curr_img, next_img, joint_diff = [b.to(self.device) for b in batch]
            with torch.no_grad():
                curr_repr = self.encoder(curr_img)
                next_repr = self.encoder(next_img)

                loss = 0
                if 'contrastive' in self.total_loss_type:
                    repr_loss = self.repr_loss_fn(
                        curr_repr,
                        next_repr
                    )
                    loss += repr_loss

                if 'joint' in self.total_loss_type:
                    all_repr = torch.concat([curr_repr, next_repr], dim=-1)
                    pred_joint_diff = self.linear_layer(all_repr)
                    joint_diff_loss = self.joint_diff_loss_fn(joint_diff, pred_joint_diff)
                    loss += joint_diff_loss * self.joint_diff_scale_factor 
                # repr_loss = self.repr_loss_fn(
                #     curr_repr,
                #     next_repr
                # )
            #     all_repr = torch.concat([curr_repr, next_repr], dim=-1)
            #     pred_joint_diff = self.linear_layer(all_repr)
            #     joint_diff_loss = self.joint_diff_loss_fn(joint_diff, pred_joint_diff)

            # loss = repr_loss + joint_diff_loss * self.joint_diff_scale_factor
            test_loss += loss.item()

        return test_loss / len(test_loader)
