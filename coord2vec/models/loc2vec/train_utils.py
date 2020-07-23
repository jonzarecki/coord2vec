import torch
import numpy as np
import os
from coord2vec.models.loc2vec.loc2vec_config import TILE_SIZE


def fit(train_loader, model, loss_fn, optimizer, scheduler, logger, n_epochs, cuda, log_interval, filename):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples:
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    log_idx = -1
    for epoch in range(n_epochs):
        # Train stage
        train_loss, num_triplets_found, log_idx = train_epoch(train_loader, model, loss_fn, optimizer, cuda,
                                                              log_interval, logger, log_idx)

        scheduler.step()
        logger.experiment.add_scalar(f'Epoch_Triplet_Loss', train_loss, epoch)

        message = '---------- '
        message += 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        message += f'\tAverage number of triplets: {int(num_triplets_found)}'
        message += ' ----------'

        print(message)
        print(filename, end='\n\n')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(model, filename)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, logger, log_idx):
    num_triplets_found = []
    temp_num_triplets_found = []
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Because of the way we generate data, each sample would actually
        #  generate 20 images. So a batch of them would have batchsize *20.
        # In our case, the data would be of shape [bs, 20, 3, IMG_SIZE, IMG_SIZE]
        # we want it to be [bs*20, 3, IMG_SIZE, IMG_SIZE]
        # similar modification for target too
        data = data.view(-1, 3, TILE_SIZE, TILE_SIZE)
        target = target.view(-1)
        if cuda:
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        outputs = model(data)
        loss_outputs = loss_fn(outputs, target)
        loss = loss_outputs[0]
        losses.append(loss.item())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        # End mixed precision training changes

        num_triplets_found.append(loss_outputs[1])
        temp_num_triplets_found.append(loss_outputs[1])

        if batch_idx % log_interval == 0:
            logger.experiment.add_scalar(f'Triplet_Loss', loss.item(), log_idx)
            logger.experiment.add_scalar(f'num_triplets_found', loss_outputs[1], log_idx)
            logger.experiment.add_scalar(f'avg_dist', loss_outputs[2], log_idx)
            # need to divide by 10 because a five-crop was done added to five center crop in geo_tile_dataset
            num_samples_has_trained = batch_idx * (data.shape[0] // 10)
            total_num_samples = len(train_loader.dataset)
            precentage_samples_trained = 100. * batch_idx / len(train_loader)
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                num_samples_has_trained, total_num_samples, precentage_samples_trained, np.mean(losses))
            message += '\t{}: {:.0f}'.format('Avg_num_triplets_found', np.mean(temp_num_triplets_found))
            dist_summary = '    '.join(['{}={:.3f}'.format(k, v) for k, v in loss_outputs[2]. items()])
            message += dist_summary
            # Reset it so that we can know intermediate
            # progress
            temp_num_triplets_found = []
            # print(loss_outputs[2])
            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, np.mean(num_triplets_found), log_idx
