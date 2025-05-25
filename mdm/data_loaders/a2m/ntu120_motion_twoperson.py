import random

import numpy as np
import torch
# from utils.action_label_to_idx import action_label_to_idx
from data_loaders.tensors import collate
from utils.misc import to_torch
import pickle
from data_loaders.a2m.imutils import flip_pose

# two-person actions in NTU120
TWO_PERSON_IDX = [49,50,51,52,53,54,55,56,57,58,59,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119]

class NTU120Motion_TP(torch.utils.data.Dataset):
    dataname = "ntu120motion"
    def __init__(self, num_frames=1, sampling="conseq", sampling_step=1, split="train", 
                 pose_rep="rot6d", translation=True, glob=True, max_len=-1, min_len=-1, num_seq_max=-1,
                 ntu_mode='xsub', num_actions=120, single_only=False, one_motion=False, pkldatafilepath=None,
                 two_only=False, flip=False, **kwargs):
        if single_only and two_only:
            raise ValueError("single_only and two_only cannot be both True")
        
        self.num_frames = num_frames
        self.sampling = "random_conseq" if split=='train' else "conseq"
        # self.sampling = sampling
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.max_len = max_len
        self.min_len = min_len
        self.num_seq_max = num_seq_max
        self.flip = flip if split == 'train' else False
        
        self.ntu_mode = ntu_mode
        self.two_person_idx = TWO_PERSON_IDX
        
        if pkldatafilepath is None:
            pkldatafilepath = f'datasets/nturgbd120/ntu120_{ntu_mode}_{split}.pkl'
        else:
            pkldatafilepath = pkldatafilepath
            
        data = pickle.load(open(pkldatafilepath, "rb"))
        all_rot6d = data["rot6d"]
        all_actions = data["y"]
        all_seq_name = data["seq"]
        
        if num_actions == 60:
            filtered_rot6d = []
            filtered_actions = []
            filtered_seq_name = []
            for rot, act, seq_name in zip(all_rot6d, all_actions, all_seq_name):
                if act < 60:
                    filtered_rot6d.append(rot)
                    filtered_actions.append(act)
                    filtered_seq_name.append(seq_name)
            all_rot6d = filtered_rot6d
            all_actions = filtered_actions
            all_seq_name = filtered_seq_name

        self._rot6d = all_rot6d
        self._actions = all_actions
        self._seq_name = all_seq_name
        
        if single_only: 
            filtered_rot6d = []
            filtered_actions = []
            filtered_seq_name = []
            for rot, act, seqn in zip(all_rot6d, all_actions, all_seq_name):
                if act not in self.two_person_idx:
                    filtered_rot6d.append(rot)
                    filtered_actions.append(act)
                    filtered_seq_name.append(seqn)
            self._rot6d = filtered_rot6d
            self._actions = filtered_actions
            self._seq_name = filtered_seq_name
        elif two_only:  #
            filtered_rot6d = []
            filtered_actions = []
            filtered_seq_name = []
            for rot, act, seqn in zip(all_rot6d, all_actions, all_seq_name):
                if act in self.two_person_idx:  #
                    filtered_rot6d.append(rot)
                    filtered_actions.append(act)
                    filtered_seq_name.append(seqn)
            self._rot6d = filtered_rot6d
            self._actions = filtered_actions
            self._seq_name = filtered_seq_name    
        else:
            self._rot6d = [x for x in all_rot6d]
            self._actions = [x for x in all_actions]
            self._seq_name = [x for x in all_seq_name]
        
        
        self._num_frames_in_video = [p.shape[-1] for p in self._rot6d]
        self.single_only = single_only
        self.two_only = two_only
        self.one_motion = one_motion
        if split == 'train':
            self._train = list(range(len(self._rot6d)))
        else:
            self._test = list(range(len(self._rot6d)))
        
        # keep_actions = np.arange(0, self.num_actions)
        # self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        # self._label_to_action = {i: x for i, x in enumerate(keep_actions)}
        # self._action_classes = ntumotion_coarse_action_enumerator
        
        if single_only:
            all_actions = np.arange(num_actions)
            mask_idx = np.setdiff1d(all_actions, self.two_person_idx)  
            self.num_actions = len(mask_idx)  

            new_enum = {}
            for new_label, old_cls in enumerate(mask_idx):
                new_enum[new_label] = ntumotion_coarse_action_enumerator[old_cls]
            self._action_classes = new_enum

            self._action_to_label = {}
            self._label_to_action = {}
            for new_label, old_cls in enumerate(mask_idx):
                self._action_to_label[old_cls] = new_label
                self._label_to_action[new_label] = old_cls
        
        elif two_only:
            mask_idx = np.array(self.two_person_idx)
            self.num_actions = len(mask_idx)

            new_enum = {}
            for new_label, old_cls in enumerate(mask_idx):
                new_enum[new_label] = ntumotion_coarse_action_enumerator[old_cls]
            self._action_classes = new_enum

            self._action_to_label = {}
            self._label_to_action = {}
            for new_label, old_cls in enumerate(mask_idx):
                self._action_to_label[old_cls] = new_label
                self._label_to_action[new_label] = old_cls
                
        else:
            self.num_actions = num_actions
            full_enum = {}
            for i in range(num_actions):
                full_enum[i] = ntumotion_coarse_action_enumerator[i]
            self._action_classes = full_enum

            self._action_to_label = {i: i for i in range(num_actions)}
            self._label_to_action = {i: i for i in range(num_actions)}
        

        self.align_pose_frontview = kwargs.get('align_pose_frontview', False)
        self.use_action_cat_as_text_labels = kwargs.get('use_action_cat_as_text_labels', False)
        self.only_60_classes = kwargs.get('only_60_classes', False)
        self.leave_out_15_classes = kwargs.get('leave_out_15_classes', False)
        self.use_only_15_classes = kwargs.get('use_only_15_classes', False)

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()

        # to remove shuffling
        self._original_train = None
        self._original_test = None

    def action_to_label(self, action):
        return self._action_to_label[action]

    def label_to_action(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            return self._label_to_action[label]
        else:  # if it is one hot vector
            label = np.argmax(label)
            return self._label_to_action[label]

    def get_pose_data(self, data_index, frame_ix):
        pose = self._load(data_index, frame_ix)
        label = self.get_label(data_index)
        return pose, label

    def get_label(self, ind):
        action = self.get_action(ind)
        return self.action_to_label(action)

    def get_action(self, ind):
        return self._actions[ind]
    
    def get_seq_name(self, ind):
        return self._seq_name[ind]

    def action_to_action_name(self, action):
        new_label = self.action_to_label(action)
        return self._action_classes[new_label]

    def action_name_to_action(self, action_name):
        # self._action_classes is either a list or a dictionary. If it's a dictionary, we 1st convert it to a list
        
        all_names = list(self._action_classes.values())
        sorter = np.argsort(all_names)    
        actions = sorter[np.searchsorted(all_names, action_name, sorter=sorter)]
        return actions
    
    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        else:
            data_index = self._test[index]
        # inp, target = self._get_item_data_index(data_index)
        # return inp, target
        return self._get_item_data_index(data_index)

    def _load(self, ind, frame_ix):
        # def _load_rot6d(self, ind, frame_ix):
        #     return self._rot6d[ind][frame_ix]
        
        action_class = self._actions[ind]
        # if action_class in self.two_person_idx and len(self._rot6d[ind])==2:
        #     person_idx = random.randint(0, 1)
        # else:
        #     person_idx = 0
        
        if self.one_motion:
            if action_class in self.two_person_idx and len(self._rot6d[ind])==2:
                person_idx = random.randint(0, 1)
            else:
                person_idx = 0
            ret = self._rot6d[ind][np.newaxis, ...] if len(self._rot6d[ind].shape) == 3 else self._rot6d[ind]
            ret = ret[person_idx, :, :, frame_ix].transpose(1, 2, 0)
        
        else:
            ret = self._rot6d[ind][np.newaxis, ...] if len(self._rot6d[ind].shape) == 3 else self._rot6d[ind]
            ret = ret[:, :, :, frame_ix]
            if ret.shape[0] == 1:   # if there is only one person (apply zero padding)
                # ret = np.concatenate([ret, np.zeros_like(ret) ], axis=0)  # zero padding
                ret = np.concatenate([ret, ret], axis=0)                    # same padding
            # else:
            #     if self.split == 'train' and random.random() < 0.5:
            #         # swap the order of the two people
            #         ret = ret[[1, 0]]
                    
                
        
        ret = to_torch(ret).contiguous()
        
        if self.flip:
            for i in range(ret.shape[0]):
                if random.random() < 0.5:
                    flipped_rot6d = flip_pose(ret[i][:24].permute(2, 0, 1))
                    ret[i] = torch.cat([flipped_rot6d.permute(1, 2, 0), ret[i][24:]], dim=0)
        
        return ret.float()

    def _get_item_data_index(self, data_index):
        nframes = self._num_frames_in_video[data_index]

        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len

            if num_frames > nframes:    # if the video is shorter than the desired length -> padding
                fair = False  # True
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(nframes),
                                               num_frames,
                                               replace=True)
                    frame_ix = sorted(choices)
                else:
                    # adding the last frame until done
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes), padding))

            elif self.sampling in ["conseq", "random_conseq"]:  # if the video is longer -> subsampling
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes),
                                           num_frames,
                                           replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")

        inp, action = self.get_pose_data(data_index, frame_ix)


        output = {'inp': inp, 'action': action}

        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))
        
        output['seq'] = self.get_seq_name(data_index)
        
        return output


    def get_mean_length_label(self, label):
        if self.num_frames != -1:
            return self.num_frames

        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        action = self.label_to_action(label)
        choices = np.argwhere(self._actions[index] == action).squeeze(1)
        lengths = self._num_frames_in_video[np.array(index)[choices]]

        if self.max_len == -1:
            return np.mean(lengths)
        else:
            # make the lengths less than max_len
            lengths[lengths > self.max_len] = self.max_len
        return np.mean(lengths)

    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train), num_seq_max)
        else:
            return min(len(self._test), num_seq_max)

    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test



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