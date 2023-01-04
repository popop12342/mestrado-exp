import torch
import torchtext

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def load_data_files():
    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    with open('../../ganbert_subj/data/subj/train_orig.txt', 'r') as f:
        for line in f:
            label, sentence = line.split('\t')
            train_sentences.append(sentence)
            train_labels.append(label)
    with open('../../ganbert_subj/data/subj/test.txt', 'r') as f:
        for line in f:
            label, sentence = line.split('\t')
            test_sentences.append(sentence)
            test_labels.append(label)
    return train_sentences, train_labels, test_sentences, test_labels

def build_vocab(train_sentences):
    def yield_tokens(sentences):
        for sen in sentences:
            yield tokenizer(sen)

    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_sentences), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def create_dataset(setnences, labels, vocab, seq_size):
    input_ids = []
    label_ids = []

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    for (text, label) in zip(setnences, labels):
        label_ids.append(label_pipeline(label))
        processed_text = text_pipeline(text)
        processed_text = [vocab['<pad>']] * (seq_size - len(processed_text)) + processed_text
        input_ids.append(processed_text)
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    label_ids = torch.tensor(label_ids)
    return torch.utils.data.TensorDataset(input_ids, label_ids)


def create_dataloaders(batch_size=8):
    train_sentences, train_labels, test_sentences, test_labels = load_data_files()
    vocab = build_vocab(train_sentences)
    seq_size = max([len(tokenizer(text)) for text in train_sentences])

    train_dataset = create_dataset(train_sentences, train_labels, vocab, seq_size)
    test_dataset = create_dataset(test_sentences, test_labels, vocab, seq_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, seq_size, vocab

