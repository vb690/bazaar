import os

from tqdm import tqdm

import numpy as np
from scipy.interpolate import griddata

from sklearn.preprocessing import KBinsDiscretizer

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib


def save_3D_animation(embeddings, emb_space_sizes, train_losses, test_losses,
                      opt_name, n_bins=10, horizon_size=10, cmap_name='jet',
                      **plotting_kwargs):
    """Utility function for visualizing the changes in weights over time in
    UMAP space. The visualization is in 3D for better appreciating the descent
    on the error surface.

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

    for i in tqdm(range(embeddings[0].shape[0])):

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

        if not os.path.exists(f'results\\3D_{opt_name}'):
            os.makedirs(f'results\\3D_{opt_name}')
        plt.savefig(f'results\\3D_{opt_name}\\{i}.png', bbox_inches='tight')
        plt.close('all')

    return None


def save_2D_animation(embeddings, target_optimizers, emb_space_sizes,
                      total_train_losses, total_test_losses,
                      n_bins=100, cmap_name='jet', **plotting_kwargs):
    """Utility function for visualizing the changes in weights over time in
    UMAP space. The visualization is in 2D for better appreciating the global
    loss surface.

        Args:
            - embeddings: list of embeddings, result of alligned UMAP
            - target_optimizers: list of strings, name of the optimizers
                considered.
            - emb_space_sizes: list of arrays, define the limits of the
                embedding space for the three layers of the MLP.
            - total_train_losses: list, training losses history.
            - total_test_losses: list, test losses.
            - n_bins: int, number of bins for discretizing the training loss.
            - cmap_name: string, name of the colormap used for representing
                the change in train losses.
            - **plotting_kwargs: keyword arguments, keyword arguments for the
                plotting function.

        Returns:
            - None
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.flatten()

    Z = np.array(total_train_losses).flatten()

    for layer, emb in enumerate(embeddings):

        x = emb[:, 0]
        y = emb[:, 1]

        xi = np.linspace(
            x.min(),
            x.max(),
            1000
        )
        yi = np.linspace(
            y.min(),
            y.max(),
            1000
        )
        x_grid, Y_grid = np.meshgrid(xi, yi)

        zi = griddata(
            (x, y),
            Z,
            (xi[None, :], yi[:, None]),
            method='linear'
        )
        zi = np.nan_to_num(zi, nan=Z.mean())

        cont = axs[layer].contourf(
            x_grid,
            Y_grid,
            zi,
            cmap=cmap_name,
            levels=n_bins,
            vmin=Z.min(),
            vmax=Z.max()
        )

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(
        cont,
        cax=cbar_ax,
        label='Training Loss'
    )

    for index, opt_name in enumerate(target_optimizers):

        print(f'Saving Optimizer {opt_name}')

        emb_size = len(total_test_losses[index])
        start = emb_size * index
        stop = start + emb_size
        embs = [emb[start:stop] for emb in embeddings]

        for ax_idx, ax in enumerate(axs):

            ax.set_title(
                f'Layer {ax_idx + 1} \
                \nOptimizer: {opt_name}'
            )
            if ax_idx == 0:
                ax.set_ylabel('Weights Space \n UMAP 2')
                ax.set_xlabel('Weights Space \n UMAP 1')
            else:
                ax.set_xlabel('Weights Space \n UMAP 1')

        for i in tqdm(range(embs[0].shape[0])):

            point_1 = axs[0].scatter(
                embs[0][i, 0],
                embs[0][i, 1],
                marker="*",
                c='white',
                edgecolor='k',
                s=60
            )
            point_2 = axs[1].scatter(
                embs[1][i, 0],
                embs[1][i, 1],
                c='white',
                marker="*",
                edgecolor='k',
                s=60
            )
            point_3 = axs[2].scatter(
                embs[2][i, 0],
                embs[2][i, 1],
                c='white',
                marker="*",
                edgecolor='k',
                s=60
            )

            if not os.path.exists(f'results\\2D_{opt_name}'):
                os.makedirs(f'results\\2D_{opt_name}')
            plt.savefig(
                f'results\\2D_{opt_name}\\{i}.png',
                bbox_inches='tight'
            )

            point_1.remove()
            point_2.remove()
            point_3.remove()

    return None
