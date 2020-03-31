import numpy as np
import torch


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