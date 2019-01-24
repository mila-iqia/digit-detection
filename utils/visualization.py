import matplotlib.pyplot as plt
from matplotlib import patches


def visualize_sample(sample, outer_bbox=None):
    '''
    Utility function to allow visualization of samples.

    Parameters
    ----------
    sample : dict
        Output of the dataloader or any of the transforms (except ToTensor).
    outer_bbox: tuple
        Tuple of coordinates (x1, x2, y1, y2) to bounding box surrounding
        all bboxes of digits in an image. Optional

    '''

    img = sample['image']
    boxes = sample['metadata']['boxes']
    labels = sample['metadata']['labels']

    # Display image
    _, ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(img)

    N = len(labels)  # Number of digits in image

    # Show individual boxes and labels
    for i in range(N):

        # Show bounding boxes
        c = ['r', 'k']
        if boxes is not None:
            x1, x2, y1, y2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=c[i % 2], facecolor='none')
            ax.add_patch(p)

            # Show Label
            caption = labels[i]
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11,
                    backgroundcolor="none")

    if outer_bbox is not None:

        x1_tot, x2_tot, y1_tot, y2_tot = outer_bbox

        p2 = patches.Rectangle((x1_tot, y1_tot),
                               x2_tot - x1_tot, y2_tot - y1_tot,
                               linewidth=2,
                               alpha=0.7, linestyle="dashed",
                               edgecolor='blue', facecolor='none')
        ax.add_patch(p2)
