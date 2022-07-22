import os, math, argparse, collections
import random
from tqdm import tqdm, trange
import numpy as np
import wandb

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from Data.vocab import load_vocab
import config as cfg
import model as bert
import data
import optimization as optim

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TODO : Making Vocab Process
""" random seed """
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    

def train_epoch(config, rank, epoch, model, criterion_lm, criterion_cls, optimizer, scheduler, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({rank}) {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.type(torch.LongTensor).to(config.device), value)
            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls, logits_lm = outputs[0], outputs[1]

            loss_cls = criterion_cls(logits_cls, labels_cls)
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

""" destroy_process_group """
def destroy_process_group():
    dist.destroy_process_group()

def train_model(rank, world_size, args):
    """Training Model

    Args:
        rank (int): Order of GPU
        world_size (int): Number of GPU
        args (dict): argparser arguments
    """
    master = (world_size == 0 or rank % world_size == 0)
    
    
    vocab = load_vocab(args.vocab) # 학습시킨 Vocab Loading
    
    config = cfg.Config.load(args.config)
    config.n_enc_vocab = len(vocab)
    config.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(config) 
    
    best_epoch, best_loss = 0, 0
    model = bert.BERTPretrain(config)
    if os.path.isfile(args.save_path):
        best_epoch, best_loss = model.bert.load(args.save_path)
        print(f"rank : {rank} load pretrain from : {args.save_path}, epoch={best_epoch}, loss={best_loss}")
        best_epoch += 1
    if 1 < args.n_gpu:
        model.to(config.device)
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model.to(config.device)
    
    # lm은 Masked Word Prediction임, cls는 Task가 NSP
    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean') 
    criterion_cls = torch.nn.CrossEntropyLoss()

    train_loader = data.build_pretrain_loader(vocab, args, epoch=best_epoch, shuffle=True)
    
    t_total = len(train_loader) * args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    # 아래의 optimizer_grouped_parmeters는 bias와 LayerNorm.weight가 포함되지 않은 model weight들은 args의 weight_decay를 따라가고, 나머지는 따라가지 않는다.
    # TODO: weight decay 설정하는 법 공부
    optimizer_grouped_parameters = [
        {'params' : [parameter_value for parameter_name, parameter_value in model.named_parameters() if not any(nd in parameter_name for nd in no_decay)], 'weight_decay' : args.weight_decay},
        {'params' : [parameter_value for parameter_name, parameter_value in model.named_parameters() if any(nd in parameter_name for nd in no_decay)], 'weight_decay' : 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # get_linear_schedule_with_warmup이 실제 BERT의 Pretraining시에 사용된 Scheduler이다.
    scheduler = optim.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    offset = best_epoch
    losses = []
    for step in trange(args.epoch, desc="Epoch"):
        epoch = step + offset
        if 0 < step:
            # 아마 아래와 같이 작성된 이유는 Large Corpus에 관련해 training을 할 때에는 기존처럼 모든 Data를 memory에 얹고 이를 index로 접근하는 것 조차 불가능함.
            # pretrain을 위한 Data Preprocess에서 다룬 것처럼 여러 파일로 쪼개진 형태로 학습을 진행하게 된다.
            # 그래서 이전 파일의 학습을 완료했다면, 이전 파일을 memory에서 삭제한 뒤, 새로운 파일을 train_loader로서 학습을 진행하게 됨.
            del train_loader 
            train_loader = data.build_pretrain_loader(vocab, args, epoch=epoch, shuffle=True) 
        loss = train_epoch(config, rank, epoch, model, criterion_lm, criterion_cls, optimizer, scheduler, train_loader) # output loss는 ndarray
        losses.append(loss)
        
        if master:
            best_epoch, best_loss = epoch, loss
            if isinstance(model, DistributedDataParallel):
                model.module.bert.save(best_epoch, best_loss, args.save_path)
            else:
                model.bert.save(best_epoch, best_loss, args.save_path)
            print(f">>>> rank: {rank} save model to {args.save_path}, epoch={best_epoch}, loss={best_loss:.3f}")
    
    print(f">>>> rank: {rank} losses: {losses}")
    
    # 학습 종료 후 분산처리 삭제(?)
    if 1 < args.n_gpu:
        destroy_process_group()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", type=str, required=False, help="Config File Path")
    parser.add_argument("--vocab", default="./Data/kowiki.model", type=str, required=False, help="Vocab File Path")
    parser.add_argument("--input", default="./Data/kowiki_bert_{}.json", type=str, required=False, help="Input File for Pretrain's Path")
    parser.add_argument("--practice", default=0, type=int, required=False, help="Practice Mode")
    parser.add_argument("--count", default=10, type=int, required=False, help="Count of Pretrain File")
    parser.add_argument("--save_path", default="./save/save_pretrain.pth", type=str, required=False, help="Config File Path")
    parser.add_argument("--epoch", default=20, type=int, required=False, help="Epoch")
    parser.add_argument("--batch", default=16, type=int, required=False, help="Batch Size")
    parser.add_argument("--gpu", default=None, type=int, required=False, help="GPU id to use.")
    parser.add_argument('--seed', type=int, default=42, required=False, help="Random seed for Initialization")
    parser.add_argument("--weight_decay", default=0, type=float, required=False, help="Weight Decay") # TODO: Weight Decay 개념 및 Defalut BERT Weight Decay 공부. 아래의 것들도 마찬가지
    parser.add_argument('--learning_rate', type=float, default=5e-5, required=False, help="Learning Rate")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, required=False, help="Adam Epsilon")
    parser.add_argument('--warmup_steps', type=float, default=0, required=False, help="Warmup Steps")
    
    
    
    args = parser.parse_args()
        
    args = parser.parse_args()
    
    # TODO: n_gpu와 args.gpu를 구분해야함.
    
    args.n_gpu =  0
    args.gpu = 0
    set_seed(args)
    
    train_model(0, args.n_gpu, args)
