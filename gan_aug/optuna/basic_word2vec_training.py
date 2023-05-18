import random
import time
import json
from typing import Dict, List
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from data_utils import format_time
from dataloader import create_word2vec_dataloaders
from discriminator import Discriminator
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

##Set random values
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

## Params
print_each_n_step = 50
num_train_epochs = 30
noise_size = 100
batch_size = 8
epsilon = 1e-8
EPOCHS = 20
lr=5e-5

if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def save_stats(stats: List[Dict], filename: str):
    filename = 'stats/{}.json'.format(filename)
    with open(filename, 'w') as json_file:
        json.dump(stats, json_file)

def train(dataset: str, num_aug: int = 0):
    train_dataloader, test_dataloader, seq_size, vocab = create_word2vec_dataloaders(dataset, device=device, num_aug=num_aug)

    model = Discriminator(None, input_size=300, vocab_size=len(vocab), padding_idx=vocab['<pad>'])
    print(model)
    print('discriminator parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if torch.cuda.is_available():
        model.cuda()

    ## Training
    training_stats = []

    model_vars = [v for v in model.parameters()]
    optimizer = torch.optim.AdamW(model_vars, lr=lr)

    for epoch_i in range(EPOCHS):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')

        t0 = time.time()
        tr_d_loss = 0

        model.train()

        for step, (text, label) in enumerate(train_dataloader):
            if step % print_each_n_step == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            
            features, logits, probs = model(text)
            logits = logits[:,0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # The discriminator provides an output for labeled and unlabeled real data
            # so the loss evaluated for unlabeled data is ignored (masked)
            label2one_hot = torch.nn.functional.one_hot(label, 2)
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            # per_example_loss = torch.masked_select(per_example_loss, offset.to(device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            d_loss = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)

            #---------------------------------
            #  OPTIMIZATION
            #---------------------------------
            # Avoid gradient accumulation
            optimizer.zero_grad()
            
            # Calculate weigth updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            d_loss.backward(retain_graph=True)
            
            # Apply modifications
            optimizer.step()
            
            # A detail log of the individual losses
            # print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}". \
            #       format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U, \
            #              g_loss_d, g_feat_reg))
            
            # Save the losses to print them 
            tr_d_loss += d_loss.item()
            
        # Calculate the average loss over all of the batches.
        avg_train_loss_d = tr_d_loss / len(train_dataloader)  
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))
        print("  Training epcoh took: {:}".format(training_time))

        # print("Saving the models...............................")
        # Saving the model
        # torch.save(transformer, 'transformer')
        # torch.save(emergency_discriminator, 'emergency_discriminator')
        # torch.save(patient_discriminator, 'patient_discriminator')

        test_accuracy = test(
            test_dataloader, model, epoch_i,
            avg_train_loss_d, training_time, training_stats
        )
        save_stats(training_stats, 'basic-1layer-{}-1aug'.format(dataset))

def test( test_dataloader: DataLoader, model: Discriminator, epoch_i: int, avg_train_loss_d: float, training_time: int, training_stats: List[Dict]):
    """Perform test step at the end of one epoch"""

    print("")
    print("Running Test...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_test_loss = 0
    all_preds = []
    all_labels_ids = []

    #loss
    nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Evaluate data for one epoch
    for text, label in test_dataloader:
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            _, logits, probs = model(text)
            filtered_logits = logits[:,0:-1]
            total_test_loss += nll_loss(filtered_logits, label)
                
        # Accumulate the predictions and the input labels
        _, preds = torch.max(filtered_logits, 1)
        all_preds += preds.detach().cpu()
        all_labels_ids += label.detach().cpu()

    # Report the final accuracy for this validation run.
    all_preds = torch.stack(all_preds).numpy()
    all_labels_ids = torch.stack(all_labels_ids).numpy()
    test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
    print("  Accuracy: {0:.3f}".format(test_accuracy))

    # Calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_loss = avg_test_loss.item()

    # Measure how long the validation run took.
    test_time = format_time(time.time() - t0)
        
    print("  Test Loss: {0:.3f}".format(avg_test_loss))
    print("  Test took: {:}".format(test_time))

    # Record all statistics from this epoch.
    training_stats.append({
        'epoch': epoch_i + 1,
        'Training Loss discriminator': avg_train_loss_d,
        'Valid. Loss': avg_test_loss,
        'Valid. Accur.': test_accuracy,
        'Valid. F1': f1_score(all_labels_ids, all_preds),
        'Valid. Recall': recall_score(all_labels_ids, all_preds),
        'Valid. Precision': precision_score(all_labels_ids, all_preds),
        'Training Time': training_time,
        'Test Time': test_time
    })
    return test_accuracy

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='subj')
    parser.add_argument('--num_aug', help='augmentation number for expading data with EDA', default=0, type=int)
    args = parser.parse_args()
    train(args.dataset, args.num_aug)