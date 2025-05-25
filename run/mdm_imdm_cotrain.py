import argparse
import numpy as np
import pickle
import os
import json
from tqdm import tqdm
import torch 
from mdm.utils.fixseed import fixseed
from mdm.utils.parser_util import train_args
from utils.parser_util import generate_args
from mdm.utils import dist_util
from mdm.train.training_loop import TrainLoop
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.data_loaders.get_data import get_dataset_loader, concat_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.diffusion.resample import create_named_schedule_sampler
from mdm.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from mdm.diffusion.fp16_util import MixedPrecisionTrainer

from functions.triplet import sample_triplets_hard_negative
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
else:  # ntu13
    num_classes = 13
    train_pkl = "dataset/ntu13_train.pkl"
    test_pkl = "dataset/ntu13_val.pkl"

traindata = get_dataset_loader(name='ntu120_tp', batch_size=batch_size, num_frames=n_frames, split='train',
                                    num_actions=num_classes, single_only=single_only, pkldatafilepath=train_pkl)
testdata = get_dataset_loader(name='ntu120_tp', batch_size=batch_size, num_frames=n_frames, split='val',
                                    num_actions=num_classes, single_only=single_only, pkldatafilepath=test_pkl)

device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


rot2xyz_pose_rep = 'rot6d'
print("creating model and diffusion...")
mdm_model, diffusion = create_model_and_diffusion(args, traindata)
mdm_model.to(device)
mdm_model.rot2xyz.smpl_model.eval()
print('Total params: %.2fM' % (sum(p.numel() for p in mdm_model.parameters_wo_clip()) / 1000000.0))
print("Training...")

trainloop = TrainLoop(args, None, mdm_model, diffusion, traindata)
trainloop.device = device


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

import logging
output_dir = f'output/{dataset_name}/i_mdm_cotrain'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logging.basicConfig(
    filename=f'{output_dir}/training.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



loss_fn_cls = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)
i_mdm_lr = 5e-4
i_mdm_optimizer = torch.optim.AdamW(i_mdm.parameters(), lr=i_mdm_lr, weight_decay=1e-4)

imdm_feat_idx = [2, 5]
lr_decay_epochs = [30]
num_epoch = 2000
mdm_eval_interval = 20
imdm_eval_interval = 1
save_interval = 50000
steps = 0


best_mdm, best_i_mdm = False, False
best_top1_acc = 0
best_motion_loss = np.inf
test_metrics = {"epochs": [], "top1_acc": [], "top5_acc": []}
lambda_triplet = 0.1
train_layer = np.array([0, 1])
for epoch in range(num_epoch):
    if epoch == 0:
        train_layer = np.array([0, 1])
    else:
        train_layer += 2
    if np.all(train_layer == np.array([8,9])):
        train_layer = np.array([0, 1])
        
    if epoch in lr_decay_epochs:
        for param_group in i_mdm_optimizer.param_groups:
            param_group['lr'] *= 0.1
        print(f"i_mdm's lr decayed to {param_group['lr']}")
        # logger.info(f"i_mdm's lr decayed to {param_group['lr']}")
    
    if epoch < 80:
        i_mdm.train()
    trainloop.ddp_model.train()
    trainloop.ddp_model.rot2xyz.smpl_model.eval()
    
    motion_loss_sum, epoch_batch_count = 0.0, 0
    total_samples, top1_acc_sum, top5_acc_sum = 0, 0, 0
    top1_acc_sum_single, top1_acc_sum_two = 0, 0
    for motion, cond in traindata:
        B = motion.shape[0]
        motion = motion.to(device)
        cond['y'] = {key: val.to(trainloop.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
        action_label = cond['y']['action'].squeeze()
        #################################### [I-MDM part] ####################################
        cls_score, i_mdm_features = i_mdm(motion, return_features=True)
        loss_cls = loss_fn_cls(cls_score, action_label)
        
        # Triplet Loss
        for feature_layer_idx in range(len(i_mdm_features)):
            if feature_layer_idx in train_layer:
                feats_layer = i_mdm_features[feature_layer_idx].mean(dim=1)  # [B, d]
                anchor, pos, neg = sample_triplets_hard_negative(feats_layer, action_label)
                loss_triplet = triplet_loss_fn(anchor, pos, neg)
                loss_cls += lambda_triplet * loss_triplet
        
        top1_acc_list = get_top1_acc(cls_score, action_label, mode='split')
        batch_top1_acc, batch_single_top1, batch_two_top1 = top1_acc_list['acc_overall'], top1_acc_list['acc_single'], top1_acc_list['acc_two']
        top5_acc_list = get_top5_acc(cls_score, action_label, mode='overall')
        batch_top5_acc = top5_acc_list['acc_overall']
        
        top1_acc_sum += batch_top1_acc * len(action_label)
        top5_acc_sum += batch_top5_acc * len(action_label)
        top1_acc_sum_single += batch_single_top1 * len(action_label)
        top1_acc_sum_two += batch_two_top1 * len(action_label)
        total_samples += len(action_label)
        
        #################################### [MDM part] ####################################
        idx1_zero_mask = (motion[:, 1] == 0).view(B, -1).all(dim=1)  # [B], bool
        chosen_idx = torch.randint(0, 2, (B,), dtype=torch.long)
        chosen_idx[idx1_zero_mask] = 0      # MDM에 사용되는 motion은 0이 아닌 것만 사용
        
        micro = motion[torch.arange(B), chosen_idx, :, :, :]
        micro_cond = cond 
        t, weights = trainloop.schedule_sampler.sample(micro.shape[0], trainloop.device)
        noise = torch.randn_like(micro)
        
        micro_cond['y']['imdm_feat'] = i_mdm_features   # I-MDM features for MDM
        micro_cond['y']['imdm_feat_idx'] = imdm_feat_idx
        # micro_cond['y']['action'] = torch.argmax(cls_score, axis=1).unsqueeze(1)
        compute_losses = functools.partial(         # return: [losses, outputs, targets]
                    trainloop.diffusion.training_losses,
                    trainloop.ddp_model,
                    micro,  # [bs, ch, image_size, image_size]
                    t,      # [bs](int) sampled timesteps
                    model_kwargs=micro_cond,
                    dataset=trainloop.data.dataset,
                    noise=noise,
                )
        mdm_losses = compute_losses()
        motion_loss = (mdm_losses["loss"] * weights).mean()

        motion_loss_sum += motion_loss.item()
        epoch_batch_count += 1
        total_loss = loss_cls + motion_loss
        
        trainloop.mp_trainer.zero_grad()
        i_mdm_optimizer.zero_grad()
        trainloop.mp_trainer.backward(total_loss)
        trainloop.mp_trainer.optimize(trainloop.opt)
        i_mdm_optimizer.step()
        
        # print(f"Batch {epoch_batch_count} / {len(traindata)}: motion_loss = {motion_loss.item():.4f}", end=' / ')
        # print(f"cls_loss = {loss_cls.item():.4f} / Top-1 acc: {batch_top1_acc:.2f} / Top-5 acc: {batch_top5_acc:.2f}")
        
    avg_top1_acc = top1_acc_sum / total_samples
    avg_top5_acc = top5_acc_sum / total_samples
    avg_top1_acc_single = top1_acc_sum_single / total_samples
    avg_top1_acc_two = top1_acc_sum_two / total_samples
    
    avg_motion_loss = motion_loss_sum / epoch_batch_count
    msg = f"[Epoch {epoch}/{num_epoch}] Training Motion Loss: {avg_motion_loss:.6f}"
    print(msg)
    # logger.info(msg)
        
    msg = f"[Epoch {epoch}/{num_epoch}] Training Top-1: {avg_top1_acc:.2f}, Top-5: {avg_top5_acc:.2f} / Single Top-1: {avg_top1_acc_single:.2f} / Two Top-1: {avg_top1_acc_two:.2f}"
    print(msg)
    # logger.info(msg)
    
    
    ######################################### [Evaluation] #########################################
    i_mdm.eval()
    trainloop.ddp_model.eval()
    motion_loss_sum, epoch_batch_count = 0.0, 0
    total_samples, top1_acc_sum, top5_acc_sum = 0, 0, 0
    top1_acc_sum_single, top1_acc_sum_two = 0, 0
    with torch.no_grad():
        for motion, cond in testdata:
            B = motion.shape[0]
            motion = motion.to(device)
            cond['y'] = {key: val.to(trainloop.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}  
            action_label = cond['y']['action'].squeeze()
            #################################### [I-MDM part] ####################################
            cls_score, i_mdm_features = i_mdm(motion, return_features=True)
            loss_cls = loss_fn_cls(cls_score, action_label)
            
            top1_acc_list = get_top1_acc(cls_score, action_label, mode='split')
            batch_top1_acc, batch_single_top1, batch_two_top1 = top1_acc_list['acc_overall'], top1_acc_list['acc_single'], top1_acc_list['acc_two']
            top5_acc_list = get_top5_acc(cls_score, action_label, mode='overall')
            batch_top5_acc = top5_acc_list['acc_overall']
            
            top1_acc_sum += batch_top1_acc * len(action_label)
            top5_acc_sum += batch_top5_acc * len(action_label)
            top1_acc_sum_single += batch_single_top1 * len(action_label)
            top1_acc_sum_two += batch_two_top1 * len(action_label)
            total_samples += len(action_label)
            
            #################################### [MDM part] ####################################
            if epoch % mdm_eval_interval == 0:    
                idx1_zero_mask = (motion[:, 1] == 0).view(B, -1).all(dim=1)  # [B], bool
                chosen_idx = torch.randint(0, 2, (B,), dtype=torch.long)
                chosen_idx[idx1_zero_mask] = 0      # MDM에 사용되는 motion은 0이 아닌 것만 사용
                
                micro = motion[torch.arange(B), chosen_idx, :, :, :]
                micro_cond = cond 
                t, weights = trainloop.schedule_sampler.sample(micro.shape[0], trainloop.device)
                noise = torch.randn_like(micro)
                
                micro_cond['y']['imdm_feat'] = i_mdm_features   # I-MDM features for MDM
                micro_cond['y']['imdm_feat_idx'] = imdm_feat_idx
                # micro_cond['y']['action'] = torch.argmax(cls_score, axis=1).unsqueeze(1)
                compute_losses = functools.partial(         # return: [losses, outputs, targets]
                            trainloop.diffusion.training_losses,
                            trainloop.ddp_model,
                            micro,  # [bs, ch, image_size, image_size]
                            t,      # [bs](int) sampled timesteps
                            model_kwargs=micro_cond,
                            dataset=trainloop.data.dataset,
                            noise=noise,
                        )
                mdm_losses = compute_losses()
                motion_loss = (mdm_losses["loss"] * weights).mean()

                motion_loss_sum += motion_loss.item()
                epoch_batch_count += 1
    
    avg_top1_acc = top1_acc_sum / total_samples
    avg_top5_acc = top5_acc_sum / total_samples
    avg_top1_acc_single = top1_acc_sum_single / total_samples
    avg_top1_acc_two = top1_acc_sum_two / total_samples
    
    if epoch % mdm_eval_interval == 0:
        avg_motion_loss = motion_loss_sum / epoch_batch_count
        if avg_motion_loss < best_motion_loss:
            best_mdm = True
            best_motion_loss = avg_motion_loss
            torch.save(mdm_model.state_dict(), f'{output_dir}/mdm_model_best.pt')

        msg = f"[Epoch {epoch}/{num_epoch}] Test Motion Loss: {avg_motion_loss:.6f} (Best: {best_mdm})"
        print(msg)
        logger.info(msg)

    if avg_top1_acc > best_top1_acc:
        best_i_mdm = True
        best_top1_acc = avg_top1_acc
        torch.save(i_mdm.state_dict(), f'{output_dir}/i_mdm_model_best.pt')
    msg = f"Test Top-1: {avg_top1_acc:.2f} / Top-5: {avg_top5_acc:.2f} / Single Top-1: {avg_top1_acc_single:.2f} / Two Top-1: {avg_top1_acc_two:.2f} (Best: {best_i_mdm})"
    print(msg)
    logger.info(msg)
    best_mdm, best_i_mdm = False, False
    
    torch.save(i_mdm.state_dict(), f'{output_dir}/i_mdm_model_last.pt')
    torch.save(trainloop.ddp_model.state_dict(), f'{output_dir}/mdm_model_last.pt')