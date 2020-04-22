import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def show_sample(rgb=None, depth=None):
    assert (
        rgb is not None or depth is not None
    ), "Can't show anything if there's nothing to show."
    if rgb is not None:
        plt.figure(figsize=(20, 8))
        plt.imshow(rgb)
        plt.axis("off")
        plt.show()
    if depth is not None:
        depth = np.array(depth) + 1e-9
        plt.figure(figsize=(20, 8))
        lognorm = colors.LogNorm(vmax=65535)
        plt.imshow(
            depth, cmap="magma_r", norm=lognorm,
        )
        plt.axis("off")
        plt.show()
