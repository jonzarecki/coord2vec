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
        # y_pred_tuples = y_pred_multi[1]
        # y_pred_tensor = torch.stack(y_pred_tuples).squeeze(2)
        # features_tensor = y_tensor.transpose(0, 1)
        embeddings = embedding.cpu()
        targets = y_tensor.cpu()
        image_data = data.cpu()
        # import pdb
        # pdb.set_trace()
        # print("hello")
        if self.all_embeddings is None:
            self.all_embeddings = embeddings
            self.all_targets = targets
            self.all_image_data = image_data
        else:
            try:
                self.all_embeddings = torch.cat([embeddings, self.all_embeddings], 0)
                self.all_targets = torch.cat([targets, self.all_targets], 0)
                self.all_image_data = torch.cat([image_data, self.all_image_data], 0)
            except:
                import pdb
                pdb.set_trace()
                print("except")

    def compute(self):
        if self.all_embeddings is None:
            raise NotComputableError('EmbeddingInfo must have at least one example before it can be computed.')
        return self.all_image_data.cpu().numpy(), self.all_embeddings.cpu().numpy(), self.all_targets.cpu().numpy()
