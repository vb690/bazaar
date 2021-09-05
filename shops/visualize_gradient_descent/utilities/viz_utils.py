import os

import numpy as np

from sklearn.preprocessing import KBinsDiscretizer

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib


def save_animation(embeddings, emb_space_sizes, train_losses, test_losses,
                   opt_name, n_bins=10, horizon_size=10, cmap_name='jet',
                   **plotting_kwargs):
    """Utility function for saving the weights changes during training in UMAP
    projection.

        Args:
            - embeddings: list of embeddings, result of alligned UMAP
            - emb_space_sizes: list of arrays, define the limits of the
                embedding space for the three layers of the MLP.
            - train_losses: list, training losses history.
            - test_losses: list, test losses.
            - opt_name: string, name of the optimizer used.
            - n_bins: int, number of bins for discretizing the training loss.
            -  horizon_size: int, maximum number of points simultaneously
                on screen.
            - cmap_name: string, name of the colormap used for representing
                the change in train losses.
            - **plotting_kwargs: keyword arguments, keyword arguments for the
                plotting function.

        Returns:
            - None
    """
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode='ordinal',
        strategy='uniform'
        )

    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = np.array(train_losses)
    colors = discretizer.fit_transform(colors.reshape(-1, 1)).flatten()
    norm = plt.Normalize(colors.min(), colors.max())

    for i in range(embeddings[0].shape[0]):

        fig, axs = plt.subplots(
            1,
            3,
            figsize=(30, 10),
            subplot_kw=dict(projection='3d')
        )

        for index, emb in enumerate(embeddings):

            min_sizes, max_sizes = emb_space_sizes[index]

            past_horizon = max(0, i - horizon_size)
            axs[index].scatter(
                emb[past_horizon:i, 0],
                emb[past_horizon:i, 1],
                train_losses[past_horizon:i],
                c=[cmap(norm(color)) for color in colors[past_horizon:i]],
                **plotting_kwargs
            )

            # PLOT ON THE 2D FACES
            axs[index].plot(
                xs=emb[past_horizon:i, 0],  # x=x
                ys=train_losses[past_horizon:i],  # y=z
                c='grey',
                zdir='y',
                zs=max_sizes[1],
                linewidth=5,
                alpha=0.25
            )
            axs[index].plot(
                xs=emb[past_horizon:i, 1],  # x=y
                ys=train_losses[past_horizon:i],  # y=z
                c='grey',
                zdir='x',
                linewidth=5,
                alpha=0.25,
                zs=min_sizes[0]
            )
            axs[index].plot(
                xs=emb[past_horizon:i, 0],  # x=x
                ys=emb[past_horizon:i, 1],  # y=y
                c='grey',
                zdir='z',
                linewidth=5,
                alpha=0.25,
                zs=min_sizes[2]
            )

            axs[index].text2D(
                0.05,
                0.95,
                f'Layer {index+1}',
                transform=axs[index].transAxes
            )
            if index == 1:
                axs[index].text2D(
                    0.5,
                    1.1,
                    f'Optimizer: {opt_name} \
                    \nTrain Loss: {round(train_losses[i], 3)} \
                    \n Test Loss: {round(test_losses[i], 3)}',
                    transform=axs[index].transAxes
                )
            elif index == 2:
                axs[index].set_xlabel('Weights Space \n UMAP 1')
                axs[index].set_ylabel('Weights Space \n UMAP 2')
                axs[index].set_zlabel('Trainining Loss')

        if not os.path.exists(f'results\\{opt_name}'):
            os.makedirs(f'results\\{opt_name}')
        plt.savefig(f'results\\{opt_name}\\{i}.png', bbox_inches='tight')
        plt.close('all')

    return None
