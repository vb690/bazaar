import pickle

import numpy as np
from scipy.interpolate import interp1d

import pandas as pd

from PIL import Image

import streamlit as st

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px


def visualize_filters(embeddings, image_index=0, time_index=0, **kwargs):
    """Visulize learned filters and return figure
    """
    plt.close('all')
    filters = embeddings[time_index][image_index, :, :, :]
    column = int(np.sqrt(filters.shape[-1]))
    fig, axs = plt.subplots(
        column,
        column,
        figsize=(10, 10),
        sharex=True,
        sharey=True
    )

    for filter_index, ax in enumerate(axs.flatten()):

        ax.imshow(
            filters[:, :, filter_index],
            vmin=0,
            vmax=1,
            **kwargs
        )
        # ax.set_title(f'Filter {filter_index} after batch {time_index}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    return fig


def visualize_embedding(embedding_df, time_index, selected_images,
                        image_index):
    """Visualize UMAP traces as 3d Plots
    """
    embedding_df = embedding_df[embedding_df['id'] <= selected_images]
    n_embeddings = embedding_df['batcn_n'].max() + 1
    n_images = embedding_df['id'].max() + 1
    classes = embedding_df[embedding_df['batcn_n'] == 0]['class'].values

    f_umap_1 = interp1d(
        embedding_df['batcn_n'][embedding_df['id'] == 0],
        embedding_df['UMAP_1'].values.reshape(n_embeddings, n_images).T,
        kind='cubic'
    )
    f_umap_2 = interp1d(
        embedding_df['batcn_n'][embedding_df['id'] == 0],
        embedding_df['UMAP_2'].values.reshape(n_embeddings, n_images).T,
        kind='cubic'
    )

    z = np.linspace(0, time_index, time_index)
    palette = px.colors.diverging.Spectral
    interpolated_traces = [f_umap_1(z), f_umap_2(z)]
    traces = []
    for image in range(n_images):

        if classes[image] == image_index:
            visible = True
        else:
            visible = 'legendonly'

        trace = go.Scatter3d(
            x=interpolated_traces[0][image],
            y=z,
            z=interpolated_traces[1][image],
            mode='lines',
            line=dict(
                color=palette[classes[image]],
                width=1.5
            ),
            opacity=1,
            legendgroup=f'Category {classes[image]}',
            name=f'Category {classes[image]}',
            visible=visible
        )
        traces.append(trace)

    names = set()
    fig_embeddings = go.Figure(data=traces)
    fig_embeddings.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name)
    )
    fig_embeddings.update_layout(
        width=1000,
        height=800,
        autosize=True,
        scene=dict(
            xaxis=dict(title='UMAP 1'),
            yaxis=dict(title='Batch Number'),
            zaxis=dict(title='UMAP 2'),

        )
    )

    return fig_embeddings


@st.cache(hash_funcs={dict: lambda _: None})
def load_data(dataset_name):
    """Load convolution filters for a given dataset
    """
    images = np.load(f'results//images//{dataset_name}.npy')

    with open(f'results//filters//{dataset_name}_conv_1.pkl', 'rb') as in_conv:
        conv_1 = pickle.load(in_conv)

    with open(f'results//filters//{dataset_name}_conv_2.pkl', 'rb') as in_conv:
        conv_2 = pickle.load(in_conv)

    embedding_df = pd.read_csv(f'results//embeddings//{dataset_name}.csv')

    return images, conv_1, conv_2, embedding_df


def get_figures(images, conv_1, conv_2, embedding_df, image_index,
                batch_number, selected_images):
    """Get all the figures objects
    """
    fig_image, ax_image = plt.subplots(
        figsize=(10, 10)
    )
    ax_image.imshow(
        images[image_index, :, :, 0],
        cmap='binary'
    )
    ax_image.set_xticks([])
    ax_image.set_yticks([])

    fig_conv_1 = visualize_filters(
            conv_1,
            image_index=image_index,
            time_index=batch_number,
            cmap='magma'
        )

    fig_conv_2 = visualize_filters(
            conv_2,
            image_index=image_index,
            time_index=batch_number,
            cmap='magma'
        )

    fig_embeddings = visualize_embedding(
        embedding_df=embedding_df,
        time_index=batch_number,
        selected_images=selected_images,
        image_index=image_index
    )

    return fig_image, fig_conv_1, fig_conv_2, fig_embeddings


def run_app():
    """Run Streamlit app
    """
    st.set_page_config(
        page_title='CNN Representations Visualizer',
        page_icon='🤖',
        layout='wide'
    )

    st.title('Convolutional Neural Network Learned Representations')
    banner = Image.open('images//header.png')
    st.image(banner, caption='LeNet-5')
    images, conv_1, conv_2, embedding_df = load_data('fashion_mnist')

    st.sidebar.title('Visualizer Parameters')
    st.sidebar.header('Select Input')
    image_index = st.sidebar.selectbox(
        'Category',
        [i for i in range(10)]
    )
    selected_images = st.sidebar.slider(
        'Images Embedded',
        min_value=1,
        max_value=1000,
        value=200
    )
    st.sidebar.header('Select Training Stage')
    batch_number = st.sidebar.slider(
        'Batch Number',
        min_value=0,
        max_value=len(conv_1) - 1,
        value=len(conv_1) - 1
    )

    fig_image, fig_conv_1, fig_conv_2, fig_embs = get_figures(
        images=images,
        conv_1=conv_1,
        conv_2=conv_2,
        embedding_df=embedding_df,
        image_index=image_index,
        batch_number=batch_number,
        selected_images=selected_images
    )
    with st.beta_expander('Convolutional Filters'):
        col1_image, col2_filters, col3_filters = st.beta_columns(3)
        col1_image.header('Input Image')
        col1_image.pyplot(fig_image)
        col2_filters.header('First Convolution')
        col2_filters.pyplot(fig_conv_1)
        col3_filters.header('Second Convolution')
        col3_filters.pyplot(fig_conv_2)

    with st.beta_expander('Learned Embedding'):
        st.header('Temporal Alligned UMAP')
        st.plotly_chart(fig_embs)


if __name__ == '__main__':
    run_app()
