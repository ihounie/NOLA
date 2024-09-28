#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import json
import math
import os, sys
import numpy as np
import itertools

import torch
import random
from torch.utils.data import DataLoader
torch.set_printoptions(threshold=100000)

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)
from optimizer import (
    create_adam_optimizer, 
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)

from data_utils import FT_Dataset
from model_lorta import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import lortalib as lorta

from exp_utils import count_trainable_parameters

import tensorly as tl

import wandb


parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], 
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder.')
        
parser.add_argument('--lora_rank', type=int, default=8, help='LoRA decomposition rank')

parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA scale')

parser.add_argument('--lora_qv', action='store_true', help="Apply LoRA to query and value")

parser.add_argument('--lora_mlp', action='store_true', help="Apply LoRA to mlp")

parser.add_argument('--qbits', type=int, default=2, help='bits for quantized aware training ')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], 
                    help='language model training objective')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')


parser.add_argument('--local-rank', type=int, default=0, help='local rank')


# Additional arguments for LorTa
parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=24, help='Number of transformer layers')
parser.add_argument('--num_modules', type=int, default=4, help='Number of modules (e.g., if finetuning Q, K, V, O) = 4')
parser.add_argument('--init_adapter', default="./checkpoints/trained_models/GPT2_M/e2e_qv_LORA_lr0.005_rank8_alpha8/model.26290.pt", help='pretrained checkpoint path')

# WandB arguments
parser.add_argument('--wandb_project', type=str, default='lorta-E2E', help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default="alelab", help='WandB entity name')
# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()        
        _optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()


def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
            loss = _loss.mean() 
            
            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model, 
    optimizer, 
    scheduler, 
    train_loader, 
    valid_loader, 
    args, 
    train_step=0, 
    epoch=0
):
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    train_loader.sampler.set_epoch(epoch)

    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)

        _lm_logits, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
        ) 

        _lm_loss = _lm_loss.mean() 

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        avg_lm_loss.update(_lm_loss.item())
        optimizer_step(
            _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
        )
        
        if train_step % args.log_interval == 0: 
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | { idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            if args.rank == 0: 
                print(log_str)
                # Log training metrics to wandb
                wandb.log({
                    'train/loss': avg_lm_loss.val,
                    'train/avg_loss': avg_lm_loss.avg,
                    'train/ppl': math.exp(avg_lm_loss.avg),
                    'lr': lr,
                    'step': train_step
                })
            log_start_time = time.time()
            avg_lm_loss.reset()
        
        if train_step % args.save_interval == 0: 
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
                print('saving checkpoint', model_path)
                torch.save({'model_state_dict': lorta.lorta_state_dict(model)}, model_path)
            distributed_sync(args)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl
                
            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

            if args.rank == 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)
                wandb.log({'valid/loss': valid_loss,'valid/ppl': valid_ppl, 'step': train_step, "valid/best_ppl": best_val_ppl})

            model.train()
            distributed_sync(args)

        if train_step == args.max_step:
            break

    if args.rank == 0:
        model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path) 
    distributed_sync(args)
    return train_step

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


if __name__ == '__main__':
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)
        # Initialize wandb
        serializable_args = {k: v for k, v in vars(args).items() if is_jsonable(v)}
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=serializable_args)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    )     
    
    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed)
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed)
    )
    

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_qv=args.lora_qv, 
            lora_mlp=args.lora_mlp,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_qv=True, 
            lora_mlp=args.lora_mlp,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_qv=args.lora_qv, 
            lora_mlp=args.lora_mlp,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        lm_net.load_weight(torch.load(args.init_checkpoint)) 
    
    if args.init_adapter is not None:
        #breakpoint()
        cp = torch.load(args.init_adapter, map_location=torch.device('cpu'))



    # Assuming 'cp' is the checkpoint loaded, and 'config' contains the model configurations
    state_dict = cp['model_state_dict']

    # Extracting model parameters from the config
    d_in = config.n_embd  # Input dimension
    d_out = config.n_embd  # Output dimension (usually same as embedding size in transformers)
    num_layers = config.n_layer
    num_heads = config.n_head
    d_out_per_head = d_out // num_heads  # Output dimension per head

    # Initialize an empty tensor for storing dWs
    dWs = torch.zeros(d_in, num_layers, 2, num_heads, d_out_per_head)

    for layer_idx in range(num_layers):
        for qv_idx, qv_name in enumerate(['query', 'value']):
            # Construct the keys for A and B matrices in the state dictionary
            A_key = f"module.transformer.h.{layer_idx}.attn.nola_{qv_name}.lora_A.weight"
            B_key = f"module.transformer.h.{layer_idx}.attn.nola_{qv_name}.lora_B.weight"

            # Extract A and B matrices
            A = state_dict[A_key]  # Shape: (rank, d_in)
            B = state_dict[B_key]  # Shape: (d_out, rank)

            # Compute the weight update dW = A.T @ B
            dW = B @ A  # Shape: (d_in, d_out)
            # Reshape dW to (d_in, num_heads, d_out_per_head)
            dW_reshaped = dW.view(d_in, num_heads, d_out_per_head)

            # Store the reshaped dW in the dWs tensor
            dWs[:, layer_idx, qv_idx, :, :] = dW_reshaped
    #breakpoint()
    T = tl.tensor(dWs)
    _, factors = tl.decomposition.CP(config.lora_rank).fit_transform(T)
    #breakpoint()
    lm_net.transformer.A.weight = torch.tensor(factors[0])
    lm_net.transformer.C_H.weight = torch.tensor(factors[1])
    lm_net.transformer.C_M.weight = torch.tensor(factors[2])
    lm_net.transformer.C_L.weight = torch.tensor(factors[3])
    lm_net.transformer.B.weight = torch.tensor(factors[4])

    # The dWs tensor now contains all computed dWs in the specified shape

    lm_net = lm_net.cuda()

    if args.lora_rank > 0:
        lorta.mark_only_lorta_as_trainable(lm_net)

    trainable_params = count_trainable_parameters(lm_net)
    print(trainable_params)
    wandb.log(trainable_params)
    optimizer = create_adam_optimizer_from_args(lm_net, args)

    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
    lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args, 
                train_step=train_step, epoch=epoch
            )
            
            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')

    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)