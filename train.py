import os, sys
import pickle
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils_lab import parse_args
from utils.get_loss import get_loss, task_aware_vib_loss
from data.data_m3care import Multimodal_EHRs
from model.MyLab3 import MyPipelineWithHierarchicalGating
from torch.nn import functional as F
from utils import metrics

'''Parameter Configuration'''

args = parse_args()
alpha = args.alpha
print('args:', args)

'''
---Loading dataset---
'''
# train_dataset = Loading_Data(args, 'train', task_name, tokenizer)
# val_dataset = Loading_Data(args, 'val', task_name, tokenizer)
# test_dataset = Loading_Data(args, 'test', task_name, tokenizer)
#
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
# val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
train_dataset, train_dataloader = Multimodal_EHRs(args, 'train')
val_dataset, val_dataloader = Multimodal_EHRs(args, 'val')
test_dataset, test_dataloader = Multimodal_EHRs(args, 'test')

# args.data_ratio = 1.0

# 
# if args.data_ratio < 1.0:
#     from torch.utils.data import Subset
#    
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
    
#    
#     total_size = len(train_dataset)
#     subset_size = int(total_size * args.data_ratio)
    
#     
#     indices = list(range(total_size))
#     random.shuffle(indices)
#     subset_indices = indices[:subset_size]
    
#     
#     train_dataset = Subset(train_dataset, subset_indices)
#    
# else:
#     train_dataset = train_dataset


from torch.utils.data import DataLoader
from data.data_m3care import collate_fn
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=args.train_batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    drop_last=True
)

num_train = len(train_dataset)
num_val = len(val_dataset)
num_test  = len(test_dataset)
print('The number of training set:', num_train)
print('The number of validation set:', num_val)
print('The number of test set:', num_test)
print('The ratio of training, validation and test set:',
      num_train /(num_train + num_val + num_test),
      num_val /(num_train + num_val + num_test),
      num_val /(num_train + num_val + num_test))

'''Checking if GPU is available'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("available device: {}".format(device))

'''Training Process'''
'''Loading Model'''
RANDOM_SEED = args.seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic =True

'''Defining Model'''
model = MyPipelineWithHierarchicalGating(
    args
)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

model.enable_gradient_monitoring()

'''Defining optimizer'''
print('Training modalities', args.modalities, len(args.modalities))


if 'notes' in args.modalities and not args.use_pretrained_emb:
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'clinical_encoder.bert' not in n]},
        {'params': [p for n, p in model.named_parameters() if 'clinical_encoder.bert' in n], 'lr': args.txt_learning_rate}
    ], lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    
    # gating_params = []
    # other_params = []
    # for name, param in model.named_parameters():
    #     if ('vib_gate' in name or 'gumbel' in name or 'gate_network' in name or 
    #         'task_prompt' in name or 'task_embedding' in name):
    #         gating_params.append(param)
    #     else:
    #         other_params.append(param)
    # optimizer = torch.optim.Adam([
    #     {'params': gating_params, 'lr': args.learning_rate * 1.5}, 
    #     {'params': other_params, 'lr': args.learning_rate}
    # ], weight_decay=args.weight_decay)

    
    
    # # for name, param in model.named_parameters():
    # #     if 'gate' in name or 'gating' in name or 'task_prompt' in name or 'task_embedding' in name or 'inter_gate_mlp' in name or 'inter_attention' in name:
    # #         gating_params.append(param)
    # #     else:
    # #         other_params.append(param)
    # # optimizer = torch.optim.Adam([
    # #     {'params': gating_params, 'lr': args.learning_rate * 2.0},  
    # #     {'params': other_params, 'lr': args.learning_rate}
    # # ], weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=args.num_train_epochs, eta_min=args.learning_rate * 0.01
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=args.learning_rate * 0.001
)
print(optimizer)

start_epoch = 0
load_checkpoint = False
if load_checkpoint:
    load_path = 'path to weight'
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['net'], strict=False)
    start_epoch = checkpoint['epoch'] + 1

'''Loss Function'''
# def get_loss(y_pred, y_true):
#     loss = nn.CrossEntropyLoss()
#     return loss(y_pred, y_true.long())

'''Start training'''
epochs = args.num_train_epochs
savemodel_name = args.output_path

best_auroc = 0
best_auprc = 0
best_f1 = 0
best_recall = 0
best_precision = 0

_best_auroc = 0
_best_auprc = 0
_best_f1 = 0
_best_recall = 0
_best_precision = 0

for each_epoch in tqdm(range(start_epoch, start_epoch + epochs)):
    epoch_loss = []
    model.train()

    for idx, batch in enumerate(tqdm(train_dataloader)):
        
        basic = batch['basic'].to(device, dtype=torch.float32)
        lab = batch['lab'].to(device, dtype=torch.float32)
        med = batch['med'].to(device, dtype=torch.float32)
        diag = batch['diag'].to(device, dtype=torch.float32)
        clinical = batch['clinical']
        masks = batch['masks'].to(device, dtype=torch.float32)
        labels = batch['label'].to(device, dtype=torch.long)
        # print(labels)

        
        if args.use_pretrained_emb:
            clinical = clinical.to(device, dtype=torch.float32)  
        # print(clinical.shape)

        optimizer.zero_grad()

        inputs = {
            'triage': basic,
                'labtest': lab,
                'medications': med,
                'diagnoses': diag,
                'notes': clinical
        }

        
        output, gate_info = model(
            inputs,
        )
        # print(gate_info['modal_gate_weights']['labtest'])

        
        total_loss, loss_info = task_aware_vib_loss(
            output, labels, gate_info, 
            alpha=args.pred_alpha, beta=args.ib_beta, lambda_sparse=args.sparse_lambda
        )

        epoch_loss.append(total_loss.cpu().detach().numpy())
        total_loss.backward()

       
        # alpha2 = 0.01
        # loss = get_loss(output, labels)
        grad_stats = model.get_gradient_statistics()
        model.print_gradient_summary()



        # epoch_loss.append(loss.cpu().detach().numpy())
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        model.zero_grad()

    epoch_loss = np.mean(epoch_loss)
    print(f'Epoch {each_epoch} Train Loss: {epoch_loss:.4f}')
    print(f'  Pred Loss: {loss_info["pred_loss"]:.4f}, IB Reg: {loss_info["ib_reg_loss"]:.4f}, Sparse Reg: {loss_info["sparse_reg_loss"]:.4f}')

    '''Validation'''
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            
            basic = batch['basic'].to(device, dtype=torch.float32)
            lab = batch['lab'].to(device, dtype=torch.float32)
            med = batch['med'].to(device, dtype=torch.float32)
            diag = batch['diag'].to(device, dtype=torch.float32)
            clinical = batch['clinical']
            masks = batch['masks'].to(device, dtype=torch.float32)
            labels = batch['label'].to(device, dtype=torch.long)

           
            if args.use_pretrained_emb:
                clinical = clinical.to(device, dtype=torch.float32)  

            inputs = {
                'triage': basic,
                'labtest': lab,
                'medications': med,
                'diagnoses': diag,
                'notes': clinical
            }

            
            output, _ = model(
                inputs,
            )

            

            y_pred += list(output.cpu().detach().numpy())
            y_true += list(labels.cpu().numpy())

        ret = metrics.print_metrics_binary(y_true, y_pred, verbose=0)

        cur_aupr = ret['auprc']
        if cur_aupr > best_auprc:
            best_auroc = ret['auroc']
            best_auprc = ret['auprc']
            best_f1 = ret['f1_score']
            best_recall = ret['rec1']
            best_precision = ret['prec1']

            auprc_state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': each_epoch
            }

            # torch.save(state, savemodel_name +'_aupr_m3care')

            print('------------ Save best model - auprc: %.4f ------------ ' %cur_aupr)
            print("precision class 1 = {}".format(best_precision))
            print("recall class 1 = {}".format(best_recall))
            print("AUC of ROC = {}".format(best_auroc))
            print("AUC of PRC = {}".format(best_auprc))
            print("f1_score = {}".format(best_f1))


        _cur_f1 = ret['f1_score']
        if _cur_f1 > _best_f1:
            _best_auroc = ret['auroc']
            _best_auprc = ret['auprc']
            _best_f1 = _cur_f1
            _best_recall = ret['rec1']
            _best_precision = ret['prec1']

            f1_state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': each_epoch
            }

            # torch.save(state, savemodel_name +'_f1_m3care')

            print('------------ Save best model - F1: %.4f ------------ ' %_cur_f1)
            print("precision class 1 = {}".format(_best_precision))
            print("recall class 1 = {}".format(_best_recall))
            print("AUC of ROC = {}".format(_best_auroc))
            print("AUC of PRC = {}".format(_best_auprc))
            print("f1_score = {}".format(_best_f1))

    # scheduler.step(ret['auprc'])
    scheduler.step()

    # if each_epoch == 1:
    #     torch.save(auprc_state, 'ihm_3.pth')
    #     print('saved!!!!!!!!!!!!!!!!')

    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {each_epoch}, current learning rate: {current_lr:.6f}')
    print('Epoch %d, roc = %.4f, prc = %.4f, f1 = %.4f, recall = %.4f, precision = %.4f' \
          % (each_epoch, ret['auroc'], ret['auprc'], ret['f1_score'], ret['rec1'], ret['prec1']))



print('t_auroc %.4f'%(best_auroc))
print('best_auprc %.4f'%(best_auprc))
print('best_f1 %.4f'%(best_f1) )
print('best_recall %.4f'%(best_recall))
print('best_precision %.4f'%(best_precision))


model.load_state_dict(auprc_state['net'], strict=False)

with torch.no_grad():
    for idx, batch in enumerate(tqdm(test_dataloader)):
        
        basic = batch['basic'].to(device, dtype=torch.float32)
        lab = batch['lab'].to(device, dtype=torch.float32)
        med = batch['med'].to(device, dtype=torch.float32)
        diag = batch['diag'].to(device, dtype=torch.float32)
        clinical = batch['clinical']
        masks = batch['masks'].to(device, dtype=torch.float32)
        labels = batch['label'].to(device, dtype=torch.long)

        
        if args.use_pretrained_emb:
            clinical = clinical.to(device, dtype=torch.float32)  # 已经是文本列表

        inputs = {
            'triage': basic,
            'labtest': lab,
            'medications': med,
            'diagnoses': diag,
            'notes': clinical
        }

       
        output, _ = model(
            inputs,
        )

        y_pred += list(output.cpu().detach().numpy())
        y_true += list(labels.cpu().numpy())

    ret = metrics.print_metrics_binary(y_true, y_pred, verbose=0)
    print('roc = %.4f, prc = %.4f, f1 = %.4f, recall = %.4f, precision = %.4f' \
          % (ret['auroc'], ret['auprc'], ret['f1_score'], ret['rec1'], ret['prec1']))
    # torch.save(model.state_dict(), 'ihm_fl.pth')
