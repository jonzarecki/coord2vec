import matplotlib.pyplot as plt
import random


def build_example_image_figure(model, features_batch, images_batch, epoch):
    fig = plt.figure(epoch, figsize=(3, 1.5), dpi=500)
    r = random.randint(0, len(images_batch) - 1)
    im = images_batch[r].cpu().numpy().swapaxes(0, 1).swapaxes(1, 2).astype('int')
    batch_output = model.forward(images_batch)[1]
    plt.axis("off")
    plt.imshow(im[:, :, 0])
    title_font = {'size': '4', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom', 'wrap': True,
                  'ha': 'left'}  # Bottom vertical alignment for more space
    fig.text(0.15, 0.75, f"actual: [{', '.join([str(a) for a in features_batch[r].cpu().numpy()])}]",
             **title_font)
    fig.text(0.15, 0.8, f"predicted: [{', '.join([str(head_out[r].item()) for head_out in batch_output])}]",
             **title_font)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(im[:, :, 1])
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(im[:, :, 2])
    return fig