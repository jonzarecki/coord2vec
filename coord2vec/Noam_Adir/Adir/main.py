import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from itertools import product
from autoencoder import Autoencoder

num_epochs = 100
num_train = 500


def add_embedding_visualization_tensor_board(model, logger, num_of_vectors=100):
    # helper function
    def select_n_random(data, labels, n=num_of_vectors):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)

        perm = np.random.permutation(len(data))

        return data[perm][:n], labels[perm][:n]

    # select random images and their target indices
    # print(model.coords_train)
    X, coords = select_n_random(model.X_train, model.coords_train.values)
    embedding = model.encoder(torch.from_numpy(X).float())
    coords = [str(coord) for coord in coords]

    # log embeddings
    logger.experiment.add_embedding(embedding, metadata=coords)


def main():
    logs_path = '/mnt/adir_logs'
    parameters = {
        'lr': [1e-1],
        'weight_decay': [1e-8],
        'emb_dim': [15],
        'batch_size': [100],
    }
    parameters_values = list(parameters.values())
    for lr, weight_decay, emb_dim, batch_size in product(*parameters_values):
        comment = f'lr={lr}, wd={weight_decay}, emb_dim={emb_dim}, bsize={batch_size}'
        logger = TensorBoardLogger(save_dir=logs_path, name=comment)
        model = Autoencoder(learning_rate=lr, weight_decay=weight_decay, num_train=num_train,
                            batch_size=batch_size, embedding_dim=emb_dim, use_all_data=False)
        trainer = Trainer(max_epochs=num_epochs, logger=logger, gpus=0)
        trainer.fit(model)
        add_embedding_visualization_tensor_board(model, logger)
        # logger.experiment.add_hparams(
        #     {'learning rate': lr, 'weight decay': weight_decay, 'embedding dim': emb_dim, 'batch size': batch_size},
        #     {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i}
        # )



main()
