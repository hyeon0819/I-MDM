import argparse
import numpy as np
import pickle
import os
import json
from tqdm import tqdm
import random
import torch 
from mdm.utils.fixseed import fixseed
from mdm.utils.parser_util import train_args
from utils.parser_util import generate_args
from mdm.utils import dist_util
from mdm.train.training_loop import TrainLoop
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.data_loaders.get_data import get_dataset_loader, concat_dataset_loader, get_collate_fn
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.diffusion.resample import create_named_schedule_sampler
from mdm.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from mdm.diffusion.fp16_util import MixedPrecisionTrainer

from functions.evaluation import get_top1_acc, get_top5_acc
import functools
import sys 
mdm_root = 'imdm/mdm'
if mdm_root not in sys.path:
    sys.path.append(mdm_root)
root = 'imdm'
if root not in sys.path:
    sys.path.append(root)
    

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["humanact12", "ntu13"])
parser.add_argument('--save_dir', type=str, default='save/custom')
parser.add_argument('--dataset', type=str, default='ntu120_tp')
parser.add_argument('--cond_mask_prob', type=float, default=0.0)
parser.add_argument('--lambda_rcxyz', type=float, default=1.0)
parser.add_argument('--lambda_vel', type=float, default=1.0)
parser.add_argument('--lambda_fc', type=float, default=1.0)

args = parser.parse_args()
dataset_name = args.dataset

single_only = True
batch_size = 512
n_frames = 64
if dataset_name == "humanact12":
    num_classes = 12
    train_pkl = "dataset/humanact12poses_train.pkl"
    test_pkl = "dataset/humanact12poses_test.pkl"
    gen_pkl = "dataset/gen_data/humanact12.pkl"
else:  # ntu13
    num_classes = 13
    train_pkl = "dataset/ntu13_train.pkl"
    test_pkl = "dataset/ntu13_val.pkl"
    gen_pkl = "dataset/gen_data/ntu13.pkl"

traindata = get_dataset_loader(name='ntu120_tp', batch_size=batch_size, num_frames=n_frames, split='train',
                                    num_actions=num_classes, single_only=single_only, pkldatafilepath=train_pkl)
testdata = get_dataset_loader(name='ntu120_tp', batch_size=batch_size, num_frames=n_frames, split='val',
                                    num_actions=num_classes, single_only=single_only, pkldatafilepath=test_pkl)


device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


rot2xyz_pose_rep = 'rot6d'
print("creating model and diffusion...")
mdm_model, diffusion = create_model_and_diffusion(args, traindata)
model_path = f'output/{args.dataset}/mdm_model_best.pt'
state_dict = torch.load(model_path, map_location='cpu')
load_model_wo_clip(mdm_model, state_dict)

if args.guidance_param != 1:
    mdm_model = ClassifierFreeSampleModel(mdm_model)   # wrapping model with the classifier-free sampler
mdm_model.to(device)
mdm_model.eval()  # disable random masking

from mdm.data_loaders.a2m.gen_ntu120_motion import NTU120Motion_GEN, GroupedBatchSampler, TWO_PERSON_IDX
from torch.utils.data import Dataset, DataLoader
ntu_gen_data = NTU120Motion_GEN(samples_per_class=230, num_frames=n_frames, num_classes=num_classes)
single_indices, two_indices = [], []
for i, cls_id in enumerate(ntu_gen_data.actions):
    if cls_id in TWO_PERSON_IDX:
        two_indices.append(i)
    else:
        single_indices.append(i)
sampler = GroupedBatchSampler(single_indices, two_indices, batch_size=args.batch_size, shuffle=True)
ntu_gen_loader = DataLoader(ntu_gen_data, batch_sampler=sampler)
sample_fn = diffusion.p_sample_loop



d_model = 512
num_layers = 2
nhead = 8
dim_feedforward = 256

from model.i_mdm.i_mdm import I_MDM_SkateTrans
i_mdm = I_MDM_SkateTrans(
                    d_model=d_model,
                    num_layers=num_layers,
                    max_frames=n_frames,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    num_actions=num_classes,
                        ).to(device)

i_mdm_model_path = 'output/'    # Path to the pretrained I-MDM model
i_mdm.load_state_dict(torch.load(i_mdm_model_path, map_location='cpu'))
i_mdm.eval()



from torch.utils.data import Subset, DataLoader
collate = get_collate_fn('ntu120_tp')
def make_batch_loader(dataset, indices):
    batch_size = len(indices)
    first_col, second_col = zip(*indices)
    all_indices = list(first_col) + list(second_col)
    subset = Subset(dataset, all_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, 
                        num_workers=8, drop_last=True, collate_fn=collate)
    return loader

from collections import defaultdict
def build_action2indices(dataset):
    action2indices = defaultdict(list)
    for i in range(len(dataset)):
        data_i = dataset[i]    # {'action': int, 'rot6d': ..., ...}
        a = data_i['action']
        action2indices[a].append(i)
    return action2indices

action2indices = build_action2indices(testdata.dataset)


def extract_imdm_features_for_batch(i_mdm, dataset, model_kwargs, device):
    batch_actions = model_kwargs['y']['action'].squeeze(1)
    pairs_of_indices = []
    for a in batch_actions:
        candidates = action2indices[a.item()]
        chosen2 = random.sample(candidates, 2)
        pairs_of_indices.append(chosen2)

    temp_loader = make_batch_loader(dataset, pairs_of_indices)
    i_mdm_features_list = []
    for motion_temp, cond_temp in temp_loader:
        motion_temp = motion_temp.to(device)
        cond_temp['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond_temp['y'].items()}
        
        _, i_mdm_features = i_mdm(motion_temp, return_features=True)
        i_mdm_features_list.append(i_mdm_features)

    avg_i_mdm_features = []
    weights = torch.tensor([0.5, 0.5])
    for i in range(len(i_mdm_features_list)):
        weighted_sum = sum(i_mdm_features_list[j][i] * weights[j] for j in range(len(i_mdm_features_list)))
        avg_i_mdm_features.append(weighted_sum)
        
    return avg_i_mdm_features



# Generate motion data
imdm_feat_idx = [2, 5]
gen_ntu_pkl_data = {'rot6d': [], 'y': [], 'seq': []}
batch_idx = 0
for model_kwargs in ntu_gen_loader:
    batch_idx += 1
    two_mask = torch.isin(model_kwargs['y']['action'], torch.tensor(TWO_PERSON_IDX)) 
    
    motion = []
    if two_mask.any().item():   # two-person action
        with torch.no_grad():
            for _ in range(2):
                avg_i_mdm_features = extract_imdm_features_for_batch(
                                        i_mdm, testdata.dataset, model_kwargs, device)
                model_kwargs['y']['imdm_feat'] = avg_i_mdm_features
                model_kwargs['y']['imdm_feat_idx'] = imdm_feat_idx
                
                sample = sample_fn(
                    mdm_model,
                    (args.batch_size, mdm_model.njoints, mdm_model.nfeats, n_frames),  # BUG FIX
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None, progress=False, dump_steps=None, 
                    noise=None, const_noise=False)
                motion.append(sample)
    else:                       # single-person action
        with torch.no_grad():
            avg_i_mdm_features = extract_imdm_features_for_batch(
                                    i_mdm, testdata.dataset, model_kwargs, device)
            model_kwargs['y']['imdm_feat'] = avg_i_mdm_features
            model_kwargs['y']['imdm_feat_idx'] = imdm_feat_idx
            
            sample = sample_fn(
                mdm_model,
                (args.batch_size, mdm_model.njoints, mdm_model.nfeats, n_frames),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None, progress=False, dump_steps=None, 
                noise=None, const_noise=False)
        motion.append(sample)
    motion = torch.stack(motion, dim=1)
    
    gen_ntu_pkl_data['rot6d'].extend(motion.detach().cpu().numpy())
    gen_ntu_pkl_data['y'].extend(model_kwargs['y']['action'].squeeze().tolist())
    gen_ntu_pkl_data['seq'].extend(['generated'] * len(motion))
    
    print(f"Generated batch: {batch_idx}/{len(ntu_gen_loader)}")

pickle.dump(gen_ntu_pkl_data, open(gen_pkl, "wb"))