import random
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
from models.text_generator import TextGenerator
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

    train_dataloader, test_dataloader, seq_size = create_bert_dataloaders(
        trial.study.user_attrs['dataset'], batch_size=batch_size, device=device,
        tokenizer=tokenizer, num_aug=trial.study.user_attrs['num_aug'])

    # Models
    num_layers = trial.study.user_attrs['num_layers']
    discriminator = BERTDiscriminator(num_layers, seq_size, device, num_labels=len(labels))
    generator = TextGenerator(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=num_layers,
        max_seq_len=seq_size,
        device=device
    )

    # print(generator)
    # print('generator parameters: ' + str(sum(p.numel() for p in generator.parameters() if p.requires_grad)))
    # print(discriminator)
    # print('discriminator parameters: ' + str(sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))

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
            # Replace noise with input tokens for generator
            src = text.clone()  # Use real data as context, or generate random input
            # Create a dummy `trg` tensor filled with zeros or random tokens
            dummy_trg = torch.zeros((batch_size, seq_size), dtype=torch.long, device=device)
            gen_out = generator(src, trg=dummy_trg)  # Generate fake text
            gen_rep = gen_out.argmax(dim=-1)  # Convert logits to token IDs

            # Augment real text with generated fake data for discriminator
            disciminator_input = torch.cat([text, gen_rep], dim=0)
            fake_input_mask = torch.ones_like(input_mask)
            fake_input_mask[gen_rep == tokenizer.pad_token_id] = 0  # Handle padding
            input_mask = torch.cat([input_mask, fake_input_mask], dim=0)

            # Forward pass through discriminator
            features, logits, probs = discriminator(disciminator_input, input_mask)

            # Split outputs for real and fake data
            split_size = text.size(0)
            D_real_logits, D_fake_logits = torch.split(logits, split_size)
            D_real_probs, D_fake_probs = torch.split(probs, split_size)

            # Calculate losses and optimize
            g_loss = -torch.mean(torch.log(1 - D_fake_probs[:, -1] + epsilon))
            d_loss = calculate_discriminator_loss(D_real_logits, D_real_probs, D_fake_probs, label, label_mask, labels)

            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            d_loss.backward(retain_graph=True)
            gen_optimizer.step()
            dis_optimizer.step()

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


def calculate_discriminator_loss(D_real_logits, D_real_probs, D_fake_probs, label, label_mask, labels):
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
    return d_loss


def test(trial: Trial, test_dataloader: DataLoader, generator: TextGenerator, discriminator: BERTDiscriminator,
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='subj')
    parser.add_argument('--study', help='optuna study name')
    parser.add_argument('--n_trials', help='number of optuna trials', default=1, type=int)
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
    study.set_user_attr('num_aug', args.num_aug)
    study.set_user_attr('train_aug', args.train_aug)
    study.set_user_attr('num_layers', args.num_layers)
    study.set_user_attr('hidden_size', args.hidden_size)
    study.optimize(objective, n_trials=args.n_trials)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
