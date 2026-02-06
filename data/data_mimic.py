import torch
import pickle
import os
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def Multimodal_EHRs(args, mode, tokenizer=None):
    task_name = args.task
    dataset = Loading_Data(args, mode, task_name, tokenizer)
    if mode == 'train':
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn,
                                drop_last=True)
    elif mode in ['val', 'test']:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn,
                                drop_last=True)
    return dataset, dataloader

class Loading_Data(Dataset):
    def __init__(self, args, mode, task_name, tokenizer, data=None):
        if data != None:
            self.data = data
        else:
            self.data = load_dataset(args, mode, debug=False)

        self.task_name = task_name
        self.dim_labtest = args.dim_labtest
        self.dim_medications = args.dim_medications
        self.dim_diagnoses = args.dim_diagnoses
        self.dim_notes = args.dim_notes
        self.tokenizer = tokenizer
        self.num_of_notes = args.num_of_notes
        self.num_of_labtests = args.num_of_labtests
        self.use_pretrained_emb = args.use_pretrained_emb  
        self.med_timestep = 24


        self.mode = mode

    def __getitem__(self, idx):
        data_details = self.data[idx]
        stay_id = data_details['stay_id']

        triage_variables = data_details['triage_variables']
        triage_variables = np.nan_to_num(triage_variables, nan=0.0)
        triage_mask = 1 if np.sum(triage_variables) > 0 else 0  

        if 'labtest' in data_details.keys():
            labtest = data_details['labtest'].astype(float)
            lab_mask = 1 if labtest.size > 0 else 0
        else:
            labtest = np.zeros((self.num_of_labtests, self.dim_labtest))
            lab_mask = 0

        if labtest.shape[0] < self.num_of_labtests:
            padding = np.zeros((self.num_of_labtests - labtest.shape[0], self.dim_labtest))
            labtest = np.concatenate((labtest, padding), axis=0)


        if 'medications' in data_details.keys():
            medication = data_details['medications']
            med_mask = 1 if medication.size > 0 else 0

            
            if medication.shape[1] != self.dim_medications:
                medication_adj = np.zeros((medication.shape[0], self.dim_medications))
                min_dim = min(medication.shape[1], self.dim_medications)
                medication_adj[:, :min_dim] = medication[:, :min_dim]
            else:
                medication_adj = medication

            
            if medication_adj.shape[0] < self.med_timestep:
                last_row = medication_adj[-1:]
                padding = np.repeat(last_row, self.med_timestep - medication_adj.shape[0], axis=0)
                medication_adj = np.concatenate((medication_adj, padding), axis=0)
            elif medication_adj.shape[0] > self.med_timestep:
                medication_adj = medication_adj[:self.med_timestep]
        else:
            medication_adj = np.zeros((self.med_timestep, self.dim_medications))
            med_mask = 0

        if 'diagnoses' in data_details.keys():
            diagnoses = data_details['diagnoses']  
            diag_mask = 1 if np.sum(diagnoses) > 0 else 0

            if diagnoses.shape[1] != self.dim_diagnoses:
                diagnoses_adj = np.zeros((1, self.dim_diagnoses))
                min_dim = min(diagnoses.shape[1], self.dim_diagnoses)
                diagnoses_adj[:, :min_dim] = diagnoses[:, :min_dim]
            else:
                diagnoses_adj = diagnoses
        else:
            diagnoses_adj = np.zeros((1, self.dim_diagnoses))
            diag_mask = 0

        if 'notes' in data_details.keys():
            if self.use_pretrained_emb:
                
                clinical_emb = data_details['notes_embedding']
                clinical_text = None
            else:
                clinical_emb = None
                clinical_text = data_details['notes']  
            
        else:
            clinical_emb = np.zeros((1, self.dim_notes)) if self.use_pretrained_emb else None
            clinical_text = [""] if not self.use_pretrained_emb else None
            

       
        triage_tensor = torch.tensor(triage_variables, dtype=torch.float)
        lab_tensor = torch.tensor(labtest, dtype=torch.float)
        med_tensor = torch.tensor(medication_adj, dtype=torch.float)
        diag_tensor = torch.tensor(diagnoses_adj, dtype=torch.float)

        if self.use_pretrained_emb:
            clinical_tensor = torch.tensor(clinical_emb, dtype=torch.float).squeeze(0)  # [dim_notes]
        else:
            clinical_tensor = clinical_text  


        if self.task_name == 'ihm':
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label * 24.0
            label = np.nan_to_num(label, nan=0.0)
            label = torch.tensor(label, dtype=torch.float32)

        return {
            'stay_id': stay_id,
            'basic': triage_tensor,        # [5]
            'lab': lab_tensor,             # [12, 59]
            'med': med_tensor,             # [12, 474]
            'diag': diag_tensor,           # [1, 15795]
            'clinical': clinical_tensor,   # [dim_notes] 
            'label': label
        }

    def __len__(self):
        return len(self.data)

def load_dataset(args, mode, debug=False):
    if mode == 'train':
        dataPath = args.train_pkl
    elif mode == 'val':
        dataPath = args.val_pkl
    else:
        dataPath = args.test_pkl
    # dataPath = args.data_pkl
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if debug:
                data = data[:100]
    return data

def collate_fn(batch):
    return {
        'basic': torch.stack([item['basic'] for item in batch]),
        'lab': torch.stack([item['lab'] for item in batch]),
        'med': torch.stack([item['med'] for item in batch]),
        'diag': torch.stack([item['diag'] for item in batch]),
        'clinical': torch.stack([item['clinical'] for item in batch]),  
        'masks': torch.stack([item['masks'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


def collate_fn2(batch):
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        if key == 'clinical' and isinstance(batch[0][key], list):
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated
