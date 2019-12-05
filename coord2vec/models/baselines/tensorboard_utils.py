import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from attr import dataclass
from torch.utils.tensorboard import SummaryWriter
from typing import List


@dataclass
class TrainExample:
    image: np.array
    predicted: float
    actual: float
    diff: float
    sum: float


def build_example_image_figure(ex: TrainExample):
    fig = plt.figure(figsize=(3, 1.5), dpi=500)
    im = ex.image.cpu().numpy().swapaxes(0, 1).swapaxes(1, 2).astype('int')
    plt.axis("off")
    title_font = {'size': '3', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom', 'wrap': True,
                  'ha': 'left'}  # Bottom vertical alignment for more space
    fig.text(0.15, 0.75, f"actual: {ex.actual}", **title_font)
    fig.text(0.15, 0.8, f"predicted: {ex.predicted}", **title_font)
    plt.subplot(1, 3, 1)
    plt.imshow(im[:, :, 0])
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(im[:, :, 1])
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(im[:, :, 2])
    return fig


def create_summary_writer(model, data_loader, log_dir, expr_name) -> SummaryWriter:
    tb_path = os.path.join(log_dir, expr_name) if expr_name == 'test' \
        else os.path.join(log_dir, expr_name, str(datetime.datetime.now()))

    writer = SummaryWriter(tb_path)
    # data_loader_iter = iter(data_loader)
    # x, y = next(data_loader_iter)
    # try:
    #     writer.add_graph(model, y)
    # except Exception as e:
    #     print("Failed to save model graph: {}".format(e))
    return writer


def add_metrics_to_tensorboard(metrics: dict, writer: SummaryWriter, feature_names: List[str], global_step: int,
                               log_str="train"):
    avg_rmse = metrics['rmse']
    for i, feat_name in enumerate(feature_names):
        writer.add_scalar(f'{feat_name} RMSE/{log_str} RMSE', avg_rmse[i], global_step=global_step)

    # add the distance correlation metrics
    # writer.add_scalar('General/Validation Distance Correlation', metrics['corr'], global_step=global_step)


def add_embedding_visualization(writer: SummaryWriter, metrics, global_step):
    all_embeddings, all_image_data, all_targets = metrics['embedding_data']
    writer.add_embedding(all_embeddings, metadata=all_targets, label_img=all_image_data,
                         global_step=global_step, tag="coord2vec")
