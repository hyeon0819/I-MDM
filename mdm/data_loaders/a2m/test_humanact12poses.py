import pickle as pkl
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import random


class HumanAct12Poses_Test(Dataset):
    dataname = "humanact12"

    def __init__(self, samples_per_class=8, num_frames=60, num_classes=12):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.total_samples = num_classes * samples_per_class
        
        actions = []
        for c in range(num_classes):
            actions += [c]*samples_per_class 
        
        random.shuffle(actions)
        self.actions = actions
        
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Return:
         mask: shape (1, 1, num_frames), bool
         lengths: shape (), long=60
         action: shape (1,)
         action_text: str
        """
        action_id = self.actions[idx]
        action_text = humanact12_coarse_action_enumerator[action_id]

        # mask, lengths
        mask = torch.ones((1, 1, self.num_frames), dtype=torch.bool)  # all True
        lengths = torch.tensor(60, dtype=torch.long)

        action_tensor = torch.tensor([action_id], dtype=torch.long)  # shape (1,)

        cond = {
            'y': {
                'mask': mask,
                'lengths': lengths,
                'action': action_tensor,
                'action_text': action_text    
            }
        }
        return cond



humanact12_coarse_action_enumerator = {
    0: "warm_up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift_dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw",
}
