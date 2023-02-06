import torch
import random
import numpy as np
import wandb

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models import EmotionClassificationModel
from utils import prepare_data, test
from freezing_utils.freezing_factory import FreezingFactory


# Train
encoders_settings = [
    'bert-base-uncased',
    'bert-base-cased-conv',
    'roberta-base',
    'roberta-base-conv',
    'xlnet-base-cased'
]

freezing_settings = [
    'none',
    '2_frozen',
    '4_frozen',
    'full'
]


def train_net(config=None):
    with wandb.init(config=config) as run:
        models_mapping = {
            'bert-base-uncased': 'bert-base-uncased',
            'bert-base-cased-conv': 'DeepPavlov/bert-base-cased-conversational',
            'roberta-base': 'roberta-base',
            'roberta-base-conv': 'vinai/bertweet-base',
            'xlnet-base-cased': 'xlnet-base-cased'
        }
        config = wandb.config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        name_str = f"{config.encoder}_{config.freezing}_seed-{config.seed}"
        run.name = name_str
        save_name = 'saving_test/' + name_str + '.pt'

        encoder_name = models_mapping[config.encoder]

        tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        num_labels, train_dataloader, test_dataloader, dev_dataloader, test_dataset = prepare_data(tokenizer,
                                                                                                   config.data,
                                                                                                   config.batch_size,
                                                                                                   max_len=128)

        model = AutoModelForSequenceClassification.from_pretrained(
            encoder_name,
            num_labels=num_labels
        )

        some_freezing_object = FreezingFactory.get_freezing_object(config.freezing, encoder_name)
        model = some_freezing_object.freeze(model)

        # if config.freezing == 'full':
        #     model = freeze_all(model)
        # elif config.freezing == '2_frozen' or config.freezing == '4_frozen':
        #     num_layers = int(config.freezing.split('_')[0])
        #     model = freeze_some_layers(model, num_layers)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        device = 'cuda'

        # TODO: differ paralleled data settings
        answer_model = EmotionClassificationModel(model, device=device, parallel=True)

        # TODO: replace patience to CONST
        patience = 3

        # TODO: differ save settings
        answer_model.train(train_dataloader, dev_dataloader, config.epochs, optimizer, save_name, save=True,
                           patience=patience)

        test_fscore = test(test_dataset, answer_model)
        wandb.log({"test F1": test_fscore})
