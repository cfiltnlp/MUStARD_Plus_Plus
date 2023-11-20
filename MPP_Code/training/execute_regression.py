
'''
The file contains code to perform hyperparameter tuning (grid-search) and training the model.
It includes log statements that get stored in logs folder.
The regression scores and corresponding hyperparamters get stored in stats folder.
It utilizes features saved in the form of pickles in the data/extracted_features folder.
The extracted_features folder contains pickles with BART, T5, RobertA for text. Use as necessary

'''



import numpy as np
import pandas as pd
import time
import sys
import random
import pickle
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader

from models import emotion_regression_model

"""
        This script is used for hyper-parameter tuning and training the model.
"""
class Tee(object):
    """
        This class is used for printing the log in file as well as on Terminal
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def seed():
    """ This method is used for seeding the code and different points"""
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """ This method is used for seeding the worker in the dataloader"""
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed()
""" argument parser is used for running the script from terminal, makes it robust """
argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--speaker", required=True,
                       help="Enter y/Y for Speaker Dependent else n/N")

argParser.add_argument("-m", "--mode", required=True,
                       help="VTA for Video, Text,  Audio repectively")
argParser.add_argument("-c", "--context", required=True,
                       help="y/Y for Context Dependent else n/N")
argParser.add_argument("-e", "--epooch", default=500, help="Number of epooch")
argParser.add_argument("-l", "--learning_rate",
                       default=0.001, help="Learning rate")
argParser.add_argument("-p", "--patience", default=5, help="Patience")
argParser.add_argument("-b", "--batch_size", default=64, help="Batch Size")
argParser.add_argument("-cr", "--classification_report", default='n',
                       help="Prints Classification report of Validation Set ")
argParser.add_argument("-gpu", "--gpu", default=0,
                       help="Which GPU to use")
argParser.add_argument("-seed", "--seed", default=42,
                       help="SEED value")
argParser.add_argument("-d", "--dropout", default=0.3,
                       help="Dropout value")

args = argParser.parse_args()

'''
Loading data. Modify the path to point to data folder
Data folder would hold the text utterances in csv files, features pre-extracted,

'''
path = "MPP_Code/data/"
mustard_input = pd.read_csv(path+'mustard_PP_utterance.csv')
print(mustard_input.columns)

temp = open(path+'extracted_features/an_merged/features_Tbart_Vkey_Audio.pickle', 'rb')
data = pickle.load(temp)

# Normalizing class
for key in list(data.keys()):
    for idx in ['cText', 'uText', 'cAudio', 'uAudio', 'cVideo', 'uVideo']:
        data[key][idx] /= np.max(abs(data[key][idx]))

# Dataset class


class ContentDataset(Dataset):

    def __init__(self, mapping, dataset, speaker_list):
        self.mapping = mapping
        self.dataset = dataset
        self.speakers_mapping = speaker_list

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        index = self.mapping.loc[idx, 'SCENE']
        data = self.dataset[index]
        aro = self.mapping.loc[idx, 'ARO']
        val = self.mapping.loc[idx, 'VAL']
        spkr = np.eye(len(self.speakers_mapping))[self.speakers_mapping.index(
            self.mapping.loc[idx, 'SPEAKER'])]

        return data['uText'], data['cText'], data['uAudio'], data['cAudio'], data['uVideo'], data['cVideo'], spkr, val, aro


device = torch.device("cuda:"+str(args.gpu))


def regression_report(t, p):
    """Args:
            t:
                True labels
            p:
                Predicted labels
        """
    dic = defaultdict(list)

    t = np.array(t)
    p = np.array(p)
    # emo_map = {} # this for class wise analysis
    # emo_map[str(-0.5*0.89999998)[:5]] = 'Ang'
    # emo_map[str(-0.89999988 * -0.40000001)[:5]] = 'Sad'
    # emo_map[str(-0.5 * 0.40000001)[:5]] = 'Fru'
    # emo_map[str(-0.55000001 * 0.64999998)[:5]] = 'Rid'
    # emo_map[str(-0.80000001 * 0.5)[:5]] = 'Dis'

    for e, i in enumerate(t):
        dic[str(i[0]*i[1])[:5]].append(e)


    """ Formating the output"""
    print('_'*73)
    print('|\t|\t\tValence\t      ||\t\tArousal\t\t|')
    print('|'+'-'*71+'|')
    print(f'|\t|   MSE\t  |   MAE   |   R2    ||   MSE   |   MAE   |   R2       | ')
    print('|'+'-'*71+'|')
    r2 = 0.0000001, 0.0000001
    for key in list(dic.keys()):
        true = t[dic[str(key)[:5]]]
        pred = p[dic[str(key)[:5]]]
        mse = mean_squared_error(true, pred, multioutput='raw_values',squared=False)
        mae = mean_absolute_error(true, pred, multioutput='raw_values')
        # r2 = r2_score(true, pred, multioutput='raw_values')
        print(
            f'|  { mse[0]:7.3f}  |{ mae[0]:7.3f}  |  { r2[0]:7.3f}||  { mse[1]:7.3f}|  { mae[1]:7.3f}|  { r2[1]:7.3f}   | ')
    print('-'*73)
    mse = mean_squared_error(t, p, multioutput='raw_values',squared=False)
    mae = mean_absolute_error(t, p, multioutput='raw_values')
    r2 = r2_score(t, p, multioutput='raw_values')
    print(
        f'|  Avg\t|{ mse[0]:7.3f}  |{ mae[0]:7.3f}  |  { r2[0]:7.3f}||  { mse[1]:7.3f}|  { mae[1]:7.3f}|  { r2[0]:7.3f}   | ')
    print('|'+'_'*71+'|')


# flag is for returning the predictions
def evaluation(loader, mod, call, report=False, flag=False):
    """Args:
            loader:
                It is the validation dataloader
            mod:
                It is the best model, which we have to evaluate
            call:
                call is the COMMAND to be executed to run the forward method of the model
                it changed as per the modality and other possible input
            report:
                It True then the classification report for the validation set is printed
            flag:
                if True the instead of evaluation metrics, method returns the calss labels
    """
    with torch.no_grad():
        pred = []
        true = []

        #     X_test.reset_index()
        total_loss = []
        criterion = nn.SmoothL1Loss(reduction='none') #Loss function for calculating validation loss
        criterion.to(device)
        seed()
        for batch in loader:
            uText = batch[0].float().to(device)
            cText = batch[1].float().to(device)
            uAudio = batch[2].float().to(device)
            cAudio = batch[3].float().to(device)
            uVideo = batch[4].float().to(device)
            cVideo = batch[5].float().to(device)
            speaker = batch[6].float().to(device)
            y_true = torch.cat(
                (batch[7].reshape(-1, 1), batch[8].reshape(-1, 1)), dim=1).float().to(device)
            del batch
            output = eval(call)
            # call is the command to be executed, sice we have different combination of input modality, this decides the input by default
            loss = criterion(output, y_true).mean(axis=0).sum()
            del uText, cText, uAudio, cAudio, uVideo, cVideo, speaker
            # with torch.cuda.device(device):
            #     torch.cuda.empty_cache()
            total_loss.append(loss)
            pred.extend(output.detach().cpu().tolist())
            true.extend(y_true.tolist())
        if flag:
            return true, pred
        if report:
            regression_report(true, pred)
        return mean_absolute_error(true, pred, multioutput='raw_values'), mean_squared_error(true, pred, multioutput='raw_values',squared=False), r2_score(true, pred, multioutput='raw_values'), sum(total_loss)/len(total_loss)


def training(mod, criterion, optimizer, call, train_loader, valid_loader, fold, e=500, patience=5, report=False):
    """Args:
            mod :
                It is the mod we have to train
            criterion :
                Loss function, her we have Cross entropy loss
            optimizer :
              object of torch.optim class
            call:
                call is the COMMAND to be excuted to run the forward method of the model
                it changed as per the modality and other possible input
            train_loader:
                It is a instance of train dataloader
            valid_loader:
                It is a instance of validation dataloader, it is given as a input to evaluation class
            fold:
                5 FOLD {0,1,2,3,4}
            e:
                maximum epoch
            patience:
                how many epoch to wait after the early stopping condition in satisfied
            report:
                It True then the classification report for the validation set is printed, it is given as a input to evaluation class
            save:
                If true then best model for each fold is saved

    """
    print("#"*90)
    mse_losses = [0]
    valid_losses = [0]
    max_ms = 100
    patience_flag = 1
    best_epooch = 0
    print(fold, e, patience)
    print(f'| epoch\t|train-loss|valid-loss| VAL-MAE|VAL-MSE| VAL-R2 ||ARO-MAE|ARO-MSE|ARO-R2 | ')

    while e > 0:
        total_loss = []
        seed()
        for batch in train_loader:
            uText = batch[0].float().to(device)
            cText = batch[1].float().to(device)
            uAudio = batch[2].float().to(device)
            cAudio = batch[3].float().to(device)
            uVideo = batch[4].float().to(device)
            cVideo = batch[5].float().to(device)
            speaker = batch[6].float().to(device)
            y_true = torch.cat(
                (batch[7].reshape(-1, 1), batch[8].reshape(-1, 1)), dim=1).float().to(device)
            del batch
            output = eval(call)
            loss = criterion(output, y_true).mean(axis=0).sum()
            del uText, cText, uAudio, cAudio, uVideo, cVideo, speaker
            # with torch.cuda.device(device):
            #     torch.cuda.empty_cache()
            optimizer.zero_grad()
            total_loss.append(loss.detach().item())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            valid_mae, valid_mse, valid_r2, valid_loss = evaluation(
                valid_loader, mod, call, report, False)
            mse_losses.append(sum(total_loss)/len(total_loss))
            valid_losses.append(valid_loss)

            e = e-1
            if max_ms > valid_loss:
                max_ms = valid_loss
                best_model = mod
                best_epooch = 500-e
                print(
                    f'|  { best_epooch}\t|{ (sum(total_loss)/len(total_loss)):7.3f}   | { valid_loss:7.3f}  |{ valid_mae[0]:7.3f} |{ valid_mse[0]:7.3f}|{ valid_r2[0]:7.3f}\t||{ valid_mae[1]:7.3f}|{ valid_mae[1]:7.3f}|{ valid_r2[1]:7.3f}| ')

            if abs(mse_losses[-2]-mse_losses[-1]) < 0.0001:

                if patience_flag == 1:
                    e = patience
                    patience_flag = 0
            else:
                patience_flag = 1
    return evaluation(valid_loader, best_model, call, report, True), best_epooch


def get_command(input_modes, context_flag, speaker_flag):
    """
        This method is used to create the COMMAND to execute the forward method of particular model,
        Depending upon the input combination
        Args:
            input_modes:
                Input Modality {VTA, VT, VA, TA, V, T, A}
            context_flag :
                If true then "with context" else "without context" 
            speaker_flag:
                if true then Speaker dependent else Speaker INdependent
    """
    if input_modes == 'VTA':
        COMMAND = "mod(**{'uA':uVideo, 'uB':uText, 'uC':uAudio"
        if context_flag == 'y':
            COMMAND += ",'cA':cVideo, 'cB':cText, 'cC':cAudio"

    elif input_modes == 'VT':
        COMMAND = "mod(**{'uA':uVideo, 'uB':uText"
        if context_flag == 'y':
            COMMAND += ",'cA':cVideo, 'cB':cText"

    elif input_modes == 'VA':
        COMMAND = "mod(**{'uA':uVideo, 'uB':uAudio"
        if context_flag == 'y':
            COMMAND += ",'cA':cVideo, 'cB':cAudio"

    elif input_modes == 'TA':
        COMMAND = "mod(**{'uA':uText, 'uB':uAudio"
        if context_flag == 'y':
            COMMAND += ",'cA':cText, 'cB':cAudio"

    elif input_modes == 'T':
        COMMAND = "mod(**{'uA':uText"
        if context_flag == 'y':
            COMMAND += ",'cA':cText"

    elif input_modes == 'V':
        COMMAND = "mod(**{'uA':uVideo"
        if context_flag == 'y':
            COMMAND += ",'cA':cVideo"

    elif input_modes == 'A':
        COMMAND = "mod(**{'uA':uAudio"
        if context_flag == 'y':
            COMMAND += ",'cA':cAudio"
    if speaker_flag == 'y':
        COMMAND += ",'speaker_embedding':speaker})"
    else:
        COMMAND += "})"

    return COMMAND


def get_model_and_parameters(args):
    """
        args is an instance of argument parser
        which will be used to 
    """
    # Here we are sortng VTA in descending order, in order to vae consistancy in the model
    input_modes = ''.join(reversed(sorted(list(args.mode.upper()))))

    parameters = {}
    MODEL_NAME = 'Speaker_'

    parameters['num_classes'] = 2

    if args.speaker.lower() == 'y':
        MODEL_NAME += 'Dependent_'
        parameters['n_speaker'] = 24
    else:
        MODEL_NAME += 'Independent_'

    if len(input_modes) == 3:
        MODEL_NAME += 'Triple_'
        parameters['input_embedding_A'] = 2048
        parameters['input_embedding_B'] = 1024
        parameters['input_embedding_C'] = 291

    elif len(input_modes) == 2:
        MODEL_NAME += 'Dual_'
        parameters['input_embedding_A'] = 2048 if input_modes[0] == 'V' else 1024
        parameters['input_embedding_B'] = 291 if input_modes[1] == 'A' else 1024
    else:
        MODEL_NAME += 'Single_'
        parameters['input_embedding_A'] = 2048 if input_modes == 'V' else 1024 if input_modes == 'T' else 291

    MODEL_NAME += 'Mode_with'
    MODEL_NAME += 'out' if args.context.lower() == 'n' else ''
    MODEL_NAME += '_Context'

    MODEL_NAME = 'emotion_regression_model.' + MODEL_NAME

    COMMAND = get_command(
        input_modes, args.context.lower(), args.speaker.lower())
    return MODEL_NAME, parameters, COMMAND


# WE ARE NOT USING THIS
with open(path+'splits.pickle', "rb") as fp:
    split = pickle.load(fp)



#just intilizing 
video_embedding_size = 2048
audio_embedding_size = 291
text_embedding_size = 1024

shared_embedding_size = 1024
projection_embedding_size = 512

epooch = args.epooch
lr = 0.001  # args.learning_rate
patience = args.patience
batch_size = 128  # args.batch_size
dropout = 0.5



# get out model name , parameters, and command as per the arguments provided in command line
MODEL_NAME, parameters, COMMAND = get_model_and_parameters(args)

parameters['shared_embedding'] = shared_embedding_size
parameters['projection_embedding'] = projection_embedding_size
parameters['dropout'] = dropout



""" This filename is used further for storing storing stats and log"""
filename = args.mode
filename += '_context_'+args.context.upper()
filename += '_speaker_'+args.speaker.upper()


""" File to store log"""
f = open('MPP_Code/log/lrec_regression/TBart_VKey_' +
         filename+'.txt', 'a+')
""" Dataframe to store stats, will be saved as CSV """
stats = pd.DataFrame(columns=['dropout', 'lr', 'batch_size', 'shared_embedding_size',
                              'projection_embedding_size', 'epoch', 'VAL-MAE', 'VAL-MSE', 'VAL-R2', 'ARO-MAE', 'ARO-MSE', 'ARO-R2'])
""" 'original' variable is mode to switch between printing area
    if we do not want to print on log file then we will use 'original'
    if we want to log then 'f'  ---> it will print on both terminal and log file
"""
original = sys.stdout

sys.stdout = Tee(sys.stdout, f)

print(MODEL_NAME.split('.')[1])

sys.stdout = original

""" since we are loading speaker name from dict 
    we are sorting them in ascending order to remove randomness and make code reproducible
"""
speaker_list = sorted(list(mustard_input.SPEAKER.value_counts().keys()))

"""
These are various combination for parameters tuning (GRID SEARCH)
"""
for dropout in [0.2,0.3,0.4]:
    for lr in [0.001,0.0001]:
        for batch_size in [64,128]:
            for shared_embedding_size, projection_embedding_size in zip([2048, 1024], [1024, 256]):
                stat = [dropout, lr, batch_size,
                        shared_embedding_size, projection_embedding_size]
                parameters['shared_embedding'] = shared_embedding_size
                parameters['projection_embedding'] = projection_embedding_size
                parameters['dropout'] = dropout

                pred_all = []
                true_all = []
                indexes = []
                types = []

                for fold in range(5):

                    """
                        for 5 FOLD cross validation 
                        we have made the stratified splits explicitly
                        this is done in order to keep consistency in different experiments (to deal with randomness)
                    """
                    train = pd.read_csv(
                        'MPP_Code/data/val_aro_splits/train_' + str(fold)+'.csv')
                    valid = pd.read_csv(
                        'MPP_Code/data/val_aro_splits/test_' + str(fold)+'.csv')
                    seed()
                    train_dataset = ContentDataset(train, data, speaker_list)
                    seed()
                    train_loader = DataLoader(
                        train_dataset, batch_size, num_workers=0, pin_memory=False, worker_init_fn=seed_worker)
                    seed()
                    valid_dataset = ContentDataset(valid, data, speaker_list)
                    seed()
                    valid_loader = DataLoader(
                        valid_dataset, batch_size, num_workers=0, pin_memory=False, worker_init_fn=seed_worker)

                    indexes.extend(valid['SCENE'].tolist())
                    types.extend(valid['SAR_T'].tolist())

                    seed()
                    mod = eval(MODEL_NAME)(**parameters)
                    mod.to(device)
                #     print(mod)
                    seed()
                    criterion = nn.SmoothL1Loss(reduction='none')
                    criterion.to(device)
                    seed()
                    optimizer = optim.Adam(
                        params=mod.parameters(), betas=(0.5, 0.99), lr=lr)

                    (true, pred), epo = training(mod=mod, criterion=criterion, optimizer=optimizer, call=COMMAND,
                                                 train_loader=train_loader, valid_loader=valid_loader, fold=fold, e=epooch, patience=patience)
                    pred_all.extend(pred)
                    true_all.extend(true)
                # training ends here


                # FOr log file emotion wisae results
                sys.stdout = Tee(sys.stdout, f)

                print(f'n_epooch:{epo} | dropout:{dropout} | lr:{lr} | batch_size:{batch_size} | shared_embedding_size:{shared_embedding_size} | projection_embedding_size:{projection_embedding_size}')
                regression_report(true_all, pred_all)
                print('-'*100)
                print('-'*100)
                sys.stdout = original
                mse = mean_absolute_error(
                    true_all, pred_all, multioutput='raw_values')
                mae = mean_squared_error(
                    true_all, pred_all, multioutput='raw_values',squared=False)
                r2 = r2_score(true_all, pred_all, multioutput='raw_values')


                # FOr stats file emotion wise results
                stat.extend(
                    [epo, mae[0], mse[0], r2[0], mae[1], mse[1], r2[1]])
                stats.loc[len(stats)] = stat
                stats.to_csv('MPP_Code/stats/lrec_regression/TBart_Vkey_' +
                             filename+'.csv', index=False)


                """ Following code can be used to save prediction """
                results = []

                # print(indexes)
                for row in zip(indexes, types, true_all, pred_all):
                    results.append(
                        [[0], row[1], row[2][0], row[3][0], row[2][1], row[3][1]])

                results = pd.DataFrame(
                    results, columns=['SCENE', 'TYPE', 'TRUE_V', 'PRED_V', 'TRUE_A', 'PRED_A'])
                results.to_csv('MPP_Code/predictions/lrec_regression/' + args.mode+'/' +
                               filename+'_'+str(int(dropout*10))+'_'+str(batch_size)+'_'+str(shared_embedding_size)+'_'+str(len(str(lr*100)))+'_'+'.csv', index=False)


                """ Following code can be used to save Sarcasm type wise analysis (will be saved in log file)"""
                sys.stdout = Tee(sys.stdout, f)

                for ty in ['PRO', 'LIK', 'ILL', 'EMB']:
                    t_t = results[results['TYPE'] == ty][[
                        'TRUE_V', 'TRUE_A']].to_numpy()
                    p_t = results[results['TYPE'] ==
                                  ty][['PRED_V', 'PRED_A']].to_numpy()

                    print("FOR :----> ", ty)
                    regression_report(t_t, p_t)
                    print('-'*100)
                print('#'*100)
                print('#'*100)
                sys.stdout = original


f.close()
