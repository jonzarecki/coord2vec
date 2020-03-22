import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from autoencoder import Autoencoder

num_epochs = 150
batch_size = 100
learning_rate = 1e-1
weight_decay = 1e-8
num_train = 500
num_features = 18
embedding_dim = 15


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
    logs_path = "/mnt/adir_logs"
    comment = f'lr={learning_rate}, weight_decay={weight_decay}, batch_size={batch_size}'
    logger = TensorBoardLogger(save_dir=logs_path, name=comment)
    model = Autoencoder(learning_rate=learning_rate, weight_decay=weight_decay, num_train=num_train,
                        num_features=num_features, batch_size=batch_size, embedding_dim=embedding_dim)
    trainer = Trainer(max_epochs=num_epochs, logger=logger)
    trainer.fit(model)
    add_embedding_visualization_tensor_board(model, logger)


main()
