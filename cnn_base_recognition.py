import torch
import torch.nn as nn
import torch.nn.functional as func

train_batch_number = 1
validate_batch_number = 1
batch_count = 0

class BaseImageRecognition(nn.Module):
    def training_step(self, batch):
        global train_batch_number
        print("Training step %i/%i" % (train_batch_number, batch_count))
        train_batch_number += 1
        images, labels = batch
        out = self(images)
        loss = func.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        global validate_batch_number
        print("Validate step %i/%i" % (validate_batch_number, batch_count))
        validate_batch_number += 1
        images, labels = batch
        out = self(images)
        loss = func.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        print("Validation epoch end")
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        print("val_loss", epoch_loss.item())
        print("val_acc", epoch_acc.item())
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

@torch.no_grad()
def evaluate(model: BaseImageRecognition, val_loader):
    model.eval()
    out = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(out)

def fit(epochs: int, lr: float, model: BaseImageRecognition, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        global batch_count
        batch_count = len(train_loader)

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history

    
