import pickle as pkl
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
import torch
import random

# two-person actions in NTU120
TWO_PERSON_IDX = [49,50,51,52,53,54,55,56,57,58,59,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119]

class NTU120Motion_GEN(Dataset):
    dataname = "ntu120motion_gen"

    def __init__(self, samples_per_class=8, num_frames=80, num_classes=120):
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
        action_text = ntumotion_coarse_action_enumerator[action_id]

        # mask, lengths
        mask = torch.ones((1, 1, self.num_frames), dtype=torch.bool)  # all True
        lengths = torch.tensor(80, dtype=torch.long)

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



ntumotion_coarse_action_enumerator = {
    0: "drink water",
    1: "eat meal/snack",
    2: "brushing teeth",
    3: "brushing hair",
    4: "drop",
    5: "pickup",
    6: "throw",
    7: "sitting down",
    8: "standing up (from sitting position)",
    9: "clapping",
    10: "reading",
    11: "writing",
    12: "tear up paper",
    13: "wear jacket",
    14: "take off jacket",
    15: "wear a shoe",
    16: "take off a shoe",
    17: "wear on glasses",
    18: "take off glasses",
    19: "put on a hat/cap",
    20: "take off a hat/cap",
    21: "cheer up",
    22: "hand waving",
    23: "kicking something",
    24: "reach into pocket",
    25: "hopping (one foot jumping)",
    26: "jump up",
    27: "make a phone call/answer phone",
    28: "playing with phone/tablet",
    29: "typing on a keyboard",
    30: "pointing to something with finger",
    31: "taking a selfie",
    32: "check time (from watch)",
    33: "rub two hands together",
    34: "nod head/bow",
    35: "shake head",
    36: "wipe face",
    37: "salute",
    38: "put the palms together",
    39: "cross hands in front (say stop)",
    40: "sneeze/cough",
    41: "staggering",
    42: "falling",
    43: "touch head (headache)",
    44: "touch chest (stomachache/heart pain)",
    45: "touch back (backache)",
    46: "touch neck (neckache)",
    47: "nausea or vomiting condition",
    48: "use a fan (with hand or paper)/feeling warm",
    49: "punching/slapping other person",
    50: "kicking other person",
    51: "pushing other person",
    52: "pat on back of other person",
    53: "point finger at the other person",
    54: "hugging other person",
    55: "giving something to other person",
    56: "touch other person's pocket",
    57: "handshaking",
    58: "walking towards each other",
    59: "walking apart from each other",
    60: "put on headphone",
    61: "take off headphone",
    62: "shoot at the basket",
    63: "bounce ball",
    64: "tennis bat swing",
    65: "juggling table tennis balls",
    66: "hush (quite)",
    67: "flick hair",
    68: "thumb up",
    69: "thumb down",
    70: "make ok sign",
    71: "make victory sign",
    72: "staple book",
    73: "counting money",
    74: "cutting nails",
    75: "cutting paper (using scissors)",
    76: "snapping fingers",
    77: "open bottle",
    78: "sniff (smell)",
    79: "squat down",
    80: "toss a coin",
    81: "fold paper",
    82: "ball up paper",
    83: "play magic cube",
    84: "apply cream on face",
    85: "apply cream on hand back",
    86: "put on bag",
    87: "take off bag",
    88: "put something into a bag",
    89: "take something out of a bag",
    90: "open a box",
    91: "move heavy objects",
    92: "shake fist",
    93: "throw up cap/hat",
    94: "hands up (both hands)",
    95: "cross arms",
    96: "arm circles",
    97: "arm swings",
    98: "running on the spot",
    99: "butt kicks (kick backward)",
    100: "cross toe touch",
    101: "side kick",
    102: "yawn",
    103: "stretch oneself",
    104: "blow nose",
    105: "hit other person with something",
    106: "wield knife towards other person",
    107: "knock over other person (hit with body)",
    108: "grab other person's stuff",
    109: "shoot at other person with a gun",
    110: "step on foot",
    111: "high-five",
    112: "cheers and drink",
    113: "carry something with other person",
    114: "take a photo of other person",
    115: "follow other person",
    116: "whisper in other person's ear",
    117: "exchange things with other person",
    118: "support somebody with hand",
    119: "finger-guessing game (playing rock-paper-scissors)"
    }




class GroupedBatchSampler(BatchSampler):
    """
    - single_person_indices와 two_person_indices를 미리 받아서
    - 배치마다 '한 종류'의 인덱스만 뽑도록 구성.
    - 예: batch_size=16이면 two-person에서 16개 뽑거나 single-person에서 16개 뽑음
    - 배치를 번갈아가며 or 무작위로 구성하도록 할 수도 있음
    """
    def __init__(self, single_indices, two_indices, batch_size, shuffle=True):
        """
        single_indices: 리스트(혹은 array) -> single-person 액션에 해당하는 dataset 인덱스
        two_indices:    리스트 -> two-person 액션에 해당하는 dataset 인덱스
        batch_size:     배치 크기
        shuffle:        섞을지 여부
        """
        
        # 두 집합 따로 보관
        self.single_indices = list(single_indices)
        self.two_indices = list(two_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 각 그룹의 길이
        self.num_single = len(self.single_indices)
        self.num_two = len(self.two_indices)

        # 배치 개수를 계산하기 위해, 
        # ex) single: 30개, batch 8 -> 30//8=3배치 + 일부 남음 -> total_batch_수 = sum of both
        self._num_batches_single = self.num_single // self.batch_size
        self._num_batches_two = self.num_two // self.batch_size

        # 총 배치 수
        self._total_batches = self._num_batches_single + self._num_batches_two

    def __iter__(self):
        # 매 epoch마다 호출

        # (1) 인덱스 섞기
        if self.shuffle:
            random.shuffle(self.single_indices)
            random.shuffle(self.two_indices)

        # (2) 배치 단위로 잘라서 list 만들기
        single_batches = []
        idx = 0
        for _ in range(self._num_batches_single):
            batch = self.single_indices[idx : idx+self.batch_size]
            idx += self.batch_size
            single_batches.append(batch)
        
        two_batches = []
        idx = 0
        for _ in range(self._num_batches_two):
            batch = self.two_indices[idx : idx+self.batch_size]
            idx += self.batch_size
            two_batches.append(batch)

        # (3) 두 그룹 배치 합치기 (원하는 방식대로 섞거나 번갈아)
        all_batches = single_batches + two_batches
        if self.shuffle:
            random.shuffle(all_batches)

        # (4) yield로 하나씩 배치를 반환
        for b in all_batches:
            yield b

    def __len__(self):
        # 총 배치 개수
        return self._total_batches