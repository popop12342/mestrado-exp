import random
import pickle
import time
from typing import Dict, List
from argparse import ArgumentParser

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from data_utils import format_time, save_stats
from dataloader import create_bert_dataloaders
from dataset_loader import dataset_loader
from optuna.trial import Trial
from torch.utils.data import DataLoader
from models.bert_discriminator import BERTDiscriminator, model_name
from models.generator import Generator
from transformers import AutoTokenizer
from util.early_stopping import EarlyStopping

# Set random values
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

# Params
print_each_n_step = 50
num_train_epochs = 50
noise_size = 1
batch_size = 8
epsilon = 1e-8
initial_temp = 1.0
anneal_rate = 0.95
min_temp = 0.1

tokenizer = AutoTokenizer.from_pretrained(model_name)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)

# If there's a GPU available...
if torch.backends.mps.is_available():
    print('Using MPS backend')
    device = torch.device('mps')
elif torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def objective(trial: Trial) -> float:
    """Objetive function of one training trial to optimize test accuracy"""
    # Load data
    labels = dataset_loader.get_labels(trial.study.user_attrs['dataset'])
    if trial.study.user_attrs['pickle_data']:
        print('Getting dataloaders from file')
        with open(trial.study.user_attrs['pickle_data'], 'rb') as pickle_file:
            train_dataloader, test_dataloader, seq_size, vocab = pickle.load(pickle_file)
    else:
        train_dataloader, test_dataloader, seq_size = create_bert_dataloaders(
            trial.study.user_attrs['dataset'], batch_size=batch_size, device=device,
            tokenizer=tokenizer, num_aug=trial.study.user_attrs['num_aug'])

    # Models
    num_layers = trial.study.user_attrs['num_layers']
    if trial.study.user_attrs['num_layers']:
        hidden_size = trial.study.user_attrs['hidden_size']
        discriminator = BERTDiscriminator(num_layers, seq_size, device, num_labels=len(labels))
        generator = Generator(num_layers=num_layers, noise_size=noise_size, output_size=tokenizer.vocab_size,
                              hidden_size=hidden_size, initial_temperature=initial_temp)
    print(generator)
    print('generator parameters: ' + str(sum(p.numel() for p in generator.parameters() if p.requires_grad)))
    print(discriminator)
    print('discriminator parameters: ' + str(sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))

    generator.to(device)
    discriminator.to(device)
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    # Training
    training_stats = []

    g_vars = [v for v in generator.parameters()]
    d_vars = [v for v in discriminator.parameters()]

    gen_optimizer = torch.optim.AdamW(g_vars, lr=5e-5)
    dis_optimizer = torch.optim.AdamW(d_vars, lr=5e-5)

    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)

    # For each epoch...
    for epoch_i in range(0, num_train_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        tr_g_loss = 0
        tr_d_loss = 0
        true_fakes = 0

        # Put the model into training mode.
        generator.train()
        discriminator.train()

        # For each batch of training data...
        for step, (text, input_mask, label, label_mask) in enumerate(train_dataloader):

            # print('text')
            # print(text)
            # print(text.shape)
            # print('input mask')
            # print(input_mask)
            # print(input_mask.shape)
            # print('label')
            # print(label)
            # print(label.shape)
            # print('label mask')
            # print(label_mask)
            # print(label_mask.shape)

            # Progress update every print_each_n_step batches.
            if step % print_each_n_step == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            noise = torch.zeros(batch_size, seq_size, noise_size, device=device).uniform_(0, 1)
            hidden = generator.initHidden(batch_size, device)
            gen_out = generator(noise, hidden)
            gen_rep = torch.argmax(gen_out, dim=2)  # converting to token
            # print('noise')
            # print(noise.shape)

            # augment text and generator fake data
            train_aug = trial.study.user_attrs['train_aug']
            # print('gen rep')
            # print(gen_rep.shape)
            # if train_aug > 0:
            #     text, label, label_mask, gen_rep, _ = augment_real_fake_tensors(
            #         text=text,
            #         label=label,
            #         gen_rep=gen_rep,
            #         vocab=vocab,
            #         train_aug=train_aug,
            #         seq_size=seq_size
            #     )

            # Generate the output of the Discriminator for real and fake data.
            # First, we put together the output of the tranformer and the generator
            # disciminator_input = torch.cat([text, gen_out], dim=0)
            disciminator_input = torch.cat([text, gen_rep], dim=0)
            # Also, join with the fake sentences mask
            fake_input_mask = torch.ones(size=input_mask.shape, device=device)
            fake_input_mask[gen_rep == 0] = 0

            input_mask = torch.cat([input_mask, fake_input_mask], dim=0)
            # Then, we select the output of the disciminator
            features, logits, probs = discriminator(disciminator_input, input_mask)
            # print('discriminator input')
            # print(disciminator_input.shape)
            # print('features')
            # print(features.shape)
            # print('logits')
            # print(logits.shape)
            # print('probs')
            # print(probs.shape)

            # Finally, we separate the discriminator's output for the real and fake
            # data
            split_size = batch_size * (train_aug + 1)
            features_list = torch.split(features, split_size)
            # Splits the tensor into chunks. Each chunk is a view of the original tensor
            D_real_features = features_list[0]
            D_fake_features = features_list[1]

            logits_list = torch.split(logits, split_size)
            D_real_logits = logits_list[0]

            probs_list = torch.split(probs, split_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]

            # Fake labels counting
            true_fakes_batch = (torch.argmax(D_fake_probs, dim=1) == len(labels)).sum().item()
            true_fakes += true_fakes_batch

            # ---------------------------------
            #  LOSS evaluation
            # ---------------------------------
            # Generator's LOSS estimation
            g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:, -1] + epsilon))
            g_feat_reg = 0 * torch.mean(
                torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2)
                )
            g_loss = g_loss_d + g_feat_reg
            # print(g_loss_d, g_feat_reg)

            # Disciminator's LOSS estimation
            logits = D_real_logits[:, 0:-1]
            log_probs = F.log_softmax(logits, dim=-1)

            # The discriminator provides an output for labeled and unlabeled real data
            # so the loss evaluated for unlabeled data is ignored (masked)
            label2one_hot = torch.nn.functional.one_hot(label, len(labels))
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, label_mask)
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            # It may be the case that a batch does not contain labeled examples,
            # so the "supervised loss" in this case is not evaluated
            if labeled_example_count == 0:
                D_L_Supervised = 0
            else:
                D_L_Supervised = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)

            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + epsilon))
            D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + epsilon))
            d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U
            # print(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U)

            # ---------------------------------
            #  OPTIMIZATION
            # ---------------------------------
            # Avoid gradient accumulation
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # Calculate weigth updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            g_loss.backward(retain_graph=True)
            d_loss.backward(retain_graph=True)

            # Apply modifications
            gen_optimizer.step()
            dis_optimizer.step()

            # A detail log of the individual losses
            # print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}". \
            #       format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U, \
            #              g_loss_d, g_feat_reg))

            # Save the losses to print them later
            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()

        update_temperature(generator, epoch_i)

        # Calculate the average loss over all of the batches.
        avg_train_loss_g = tr_g_loss / len(train_dataloader)
        avg_train_loss_d = tr_d_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
        print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))
        print("  Training epoch took: {:}".format(training_time))
        print("  Fakes correct discriminared: {}".format(true_fakes))

        print("Saving the models...............................")
        # Saving the model
        torch.save(generator, '../models/generator')
        torch.save(discriminator, '../models/discriminator')

        test_accuracy = test(
            trial, test_dataloader, generator, discriminator, epoch_i,
            avg_train_loss_g, avg_train_loss_d, training_time, training_stats
        )
        training_stats[-1]['True fakes'] = true_fakes

        save_stats(training_stats, trial)

        # check early stopping
        early_stopping(test_accuracy)
        if early_stopping.early_stop:
            print('early stopping. Training Stopped')
            break

    return test_accuracy


def test(trial: Trial, test_dataloader: DataLoader, generator: Generator, discriminator: BERTDiscriminator,
         epoch_i: int, avg_train_loss_g: float, avg_train_loss_d: float, training_time: int,
         training_stats: List[Dict]):
    """Perform test step at the end of one epoch"""

    print("")
    print("Running Test...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    discriminator.eval()
    generator.eval()

    # Tracking variables
    total_test_loss = 0
    all_preds = []
    all_labels_ids = []

    # loss
    nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Evaluate data for one epoch
    for text, input_mask, label, label_mask in test_dataloader:
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            _, logits, probs = discriminator(text, input_mask)
            filtered_logits = logits[:, 0:-1]
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
        'Training Loss generator': avg_train_loss_g,
        'Training Loss discriminator': avg_train_loss_d,
        'Valid. Loss': avg_test_loss,
        'Valid. Accur.': test_accuracy,
        # 'Valid. F1': f1_score(all_labels_ids, all_preds),
        # 'Valid. Recall': recall_score(all_labels_ids, all_preds),
        # 'Valid. Precision': precision_score(all_labels_ids, all_preds),
        'Training Time': training_time,
        'Test Time': test_time
    })
    trial.report(test_accuracy, step=epoch_i+1)
    return test_accuracy


def update_temperature(generator, epoch):
    temperature = max(initial_temp * (anneal_rate ** epoch), min_temp)
    generator.set_temperature(temperature)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='subj')
    parser.add_argument('--pickle_data', help='pikled file contained dataloaders to skip data preparation,\
                        if not provided will create the dataloaders as usual')
    parser.add_argument('--study', help='optuna study name')
    parser.add_argument('--num_aug', help='augmentation number for expading data with EDA', default=0, type=int)
    parser.add_argument('--num_layers', help='number of layers for generator and discriminator', default=1, type=int)
    parser.add_argument('--hidden_size', help='hidden size for generator and discriminator', default=1, type=int)
    parser.add_argument('--train_aug', help='augmentation rate for expading data inside training', default=0, type=int)
    args = parser.parse_args()
    study = optuna.create_study(
        storage='sqlite:///db.sqlite3',
        study_name=args.study,
        direction='maximize',
        load_if_exists=True
    )
    study.set_user_attr('dataset', args.dataset)
    study.set_user_attr('pickle_data', args.pickle_data)
    study.set_user_attr('num_aug', args.num_aug)
    study.set_user_attr('train_aug', args.train_aug)
    study.set_user_attr('num_layers', args.num_layers)
    study.set_user_attr('hidden_size', args.hidden_size)
    study.optimize(objective, n_trials=1)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
