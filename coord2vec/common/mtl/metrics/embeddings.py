import torch
from ignite.exceptions import NotComputableError

from coord2vec.common.mtl.metrics.mtl_metric import MtlMetric


class EmbeddingData(MtlMetric):
    """
    Keeps the embeddings outputted by the network

    - `update` must receive output of the form `(y_pred, y)`.
    """

    def reset(self):
        self.all_embeddings = None
        self.all_targets = None
        self.all_image_data = None

    def update_mtl(self, data, embedding, loss, multi_losses, y_pred_tensor, y_tensor):
        # tensorboard cannot eat too much data, fails for large sprite
        if self.all_embeddings is not None and self.all_embeddings.shape[0] > 1000:
            return

        embeddings = embedding.cpu()
        targets = [str(l) for l in y_tensor.tolist()]
        image_data = data.cpu() / 255  # expects images in float
        if self.all_embeddings is None:
            self.all_embeddings = embeddings
            self.all_targets = targets
            self.all_image_data = image_data
        else:
            self.all_embeddings = torch.cat([embeddings, self.all_embeddings], 0)
            self.all_targets = self.all_targets + targets
            self.all_image_data = torch.cat([image_data, self.all_image_data], 0)

    def compute(self):
        if self.all_embeddings is None:
            raise NotComputableError('EmbeddingInfo must have at least one example before it can be computed.')
        return self.all_embeddings.cpu(), self.all_image_data.cpu(), self.all_targets
