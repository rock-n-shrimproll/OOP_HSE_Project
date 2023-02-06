import torch
import wandb
from sklearn.metrics import f1_score
from torch.nn import DataParallel


class EmotionClassificationModel:

    def __init__(self, model, device, parallel=False):
        self.model = model
        self.device = device
        self.parallel = parallel
        if self.parallel:
            self.model = DataParallel(self.model).to(device)
        else:
            self.model.to(device)

    def __call__(self, input_ids, attn_mask):

        self.model.eval()

        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = output['logits']
            if self.parallel:
                active_logits = logits.view(-1, self.model.module.num_labels)
            else:
                active_logits = logits.view(-1, self.model.num_labels)
            pred = torch.argmax(active_logits, axis=1)
        return pred

    def validate(self, val_dataloader):

        self.model.eval()

        val_loss, val_fscore = 0, 0

        with torch.no_grad():
            for batch in val_dataloader:
                tokens, labels = batch
                ids = tokens['input_ids'].to(self.device).squeeze(dim=1)
                mask = tokens['attention_mask'].to(self.device).squeeze(dim=1)
                labels = labels.to(self.device)

                output = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output['loss']
                logits = output['logits']

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                val_loss += loss.item()

                gold = labels.view(-1)
                if self.parallel:
                    active_logits = logits.view(-1, self.model.module.num_labels)
                else:
                    active_logits = logits.view(-1, self.model.num_labels)
                pred = torch.argmax(active_logits, axis=1)

                fscore = f1_score(gold.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                val_fscore += fscore

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_f1 = val_fscore / len(val_dataloader)
        return avg_val_loss, avg_val_f1

    def train(self, train_dataloader, val_dataloader, n_epoch, optimizer, model_save_name, save, patience=3):
        # TODO: rethink auto save parameter
        train_losses, val_losses = [], []
        train_fscores, val_fscores = [], []
        val_scores = []
        no_improv_epochs = 0

        for epoch in range(n_epoch):

            self.model.train()

            train_loss, train_fscore = 0, 0
            for step_num, batch in enumerate(train_dataloader):
                tokens, labels = batch
                ids = tokens['input_ids'].to(self.device).squeeze(dim=1)
                mask = tokens['attention_mask'].to(self.device).squeeze(dim=1)
                labels = labels.to(self.device)

                output = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output['loss']
                logits = output['logits']

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                gold = labels.view(-1)
                if self.parallel:
                    active_logits = logits.view(-1, self.model.module.num_labels)
                else:
                    active_logits = logits.view(-1, self.model.num_labels)
                pred = torch.argmax(active_logits, axis=1)

                fscore = f1_score(gold.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                train_fscore += fscore

                wandb.log({"batch train loss": loss.item()})

            avg_train_loss = train_loss / len(train_dataloader)
            avg_train_f1 = train_fscore / len(train_dataloader)

            # EARLY STOPPING CODE
            avg_val_loss, avg_val_f1 = self.validate(val_dataloader)
            val_scores.append(avg_val_f1)
            if max(val_scores) > avg_val_f1:
                no_improv_epochs += 1
            else:
                if save:
                    if self.parallel:
                        torch.save(self.model.module.state_dict(), model_save_name)
                    else:
                        torch.save(self.model.state_dict(), model_save_name)

            if no_improv_epochs >= patience:
                return None

            train_losses.append(avg_train_loss)
            train_fscores.append(avg_train_f1)
            val_losses.append(avg_val_loss)
            val_fscores.append(avg_val_f1)

            wandb.log({"train loss": avg_train_loss,
                       "val loss": avg_val_loss,
                       "train F1": avg_train_f1,
                       "val F1": avg_val_f1,
                       "epoch": epoch})
        return None
