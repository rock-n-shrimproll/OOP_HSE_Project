import pandas as pd
from torch.utils.data import DataLoader
from datasets import EmotionLinesDataset
from sklearn.metrics import f1_score


def prepare_data(tokenizer, data_path, bs, max_len):
    """
    Prepares datasets and data loaders from .csv files.
    Returns number of labels in data, three data loaders and test dataset (for testing).
    #TODO: improve to have a prettier return
    ----------
    tokenizer
        Tokenizer for the selected model.
    data_path : str
        Path to the directory with three data files (train, test and dev).
    bs: int
        Batch size.
    max_len: int
        Maximum sequence length. Used in tokenizer to truncate/pad sequences.
    """
    train_data = pd.read_csv(data_path + '/MeldCSV/train.csv')
    test_data = pd.read_csv(data_path + '/MeldCSV/test.csv')
    dev_data = pd.read_csv(data_path + '/MeldCSV/dev.csv')

    num_labels = len(set(train_data['emotion']))
    labels = sorted(list(set(train_data['emotion'])))
    label_dict = {}
    for i in range(len(labels)):
        label_dict[labels[i]] = i

    train_dataset = EmotionLinesDataset(train_data, label_dict, tokenizer, max_len=max_len)
    dev_dataset = EmotionLinesDataset(dev_data, label_dict, tokenizer, max_len=max_len)
    test_dataset = EmotionLinesDataset(test_data, label_dict, tokenizer, max_len=max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
    return num_labels, train_dataloader, test_dataloader, dev_dataloader, test_dataset


def test(test_dataset, answer_model):
    """
    Evaluates a model on a test set.
    ----------
    test dataset: EmotionLinesDataset
        Data as an EmotionLinesDataset object.
    answer_model
        Model to test. An EmotionClassificationModel object.
    """
    gold, pred = [], []
    for i in range(len(test_dataset)):
        res = answer_model(test_dataset[i][0]['input_ids'], test_dataset[i][0]['attention_mask'])
        pred.append(int(res))
        gold.append(test_dataset[i][1])
    fscore = f1_score(gold, pred, average='weighted')
    return fscore


def freeze_some_layers(model, num_layers):
    """
    Freezes model layers except for the last two or four and the classifier head.
    For now only works with BERT.
    """
    if num_layers == 2:
        for name, param in model.named_parameters():
            if name.startswith("classifier") or name.startswith("bert.encoder.layer.11") or name.startswith(
                    "bert.encoder.layer.10"):
                continue
            else:
                param.requires_grad = False
    elif num_layers == 4:
        for name, param in model.named_parameters():
            if name.startswith("classifier") or name.startswith("bert.encoder.layer.11") or name.startswith(
                    "bert.encoder.layer.10") or name.startswith("bert.encoder.layer.9") or name.startswith(
                    "bert.encoder.layer.8"):
                continue
            else:
                param.requires_grad = False
    else:
        print("Can't freeze this number of layers! Change the function, you lazy ass")
    return model
