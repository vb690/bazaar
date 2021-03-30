import pickle

import random

from tqdm import tqdm

import numpy as np

import pandas as pd

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.datasets import fashion_mnist

from umap.aligned_umap import AlignedUMAP


def create_model_encoders(X, y):
    """Create a CNN model and realtive encoders
    """
    inp = Input((X.shape[1], X.shape[2], X.shape[3]))

    conv_1 = Conv2D(
        filters=4,
        kernel_size=5
    )(inp)
    conv_1 = Activation('sigmoid')(conv_1)
    pol_1 = MaxPooling2D(
        pool_size=2,
        strides=(2, 2)
    )(conv_1)

    conv_2 = Conv2D(
        filters=9,
        kernel_size=5
    )(pol_1)
    conv_2 = Activation('sigmoid')(conv_2)
    pol_2 = MaxPooling2D(
        pool_size=2,
        strides=(2, 2)
    )(conv_2)

    flat = Flatten()(pol_2)
    dense = Dense(80)(flat)
    dense = Activation('sigmoid')(dense)
    dense = Dense(40)(dense)
    dense = Activation('sigmoid')(dense)
    dense = Dense(20)(dense)
    dense = Activation('sigmoid')(dense)

    out = Dense(y.max())(dense)
    out = Activation('softmax')(dense)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy'
    )

    conv_1_enc = Model(inp, conv_1)

    conv_2_enc = Model(inp, conv_2)

    dense_enc = Model(inp, dense)

    return model, conv_1_enc, conv_2_enc, dense_enc


def prepare_alligned_umap_data(embeddings, max_time_index=10):
    """Prepare data for running alligned umap
    """
    list_embeddings = []
    mappers = []
    for time_index in range(max_time_index):

        time_embedding = embeddings[time_index]
        list_embeddings.append(
            time_embedding
        )
        mappers.append(
            {embedding_index: embedding_index for
                embedding_index in range(time_embedding.shape[0])}
        )

    return list_embeddings, mappers[1:]


def load_data(loader=fashion_mnist, channels=1, batch_size=500, sample=1000):
    """Load data from dataset loader
    """
    train, test = loader.load_data()

    X_tr, y_tr = train
    X_ts, y_ts = test

    rows, columns = X_tr.shape[1], X_tr.shape[2]

    X_tr = X_tr.reshape(-1, rows, columns, channels)
    X_tr = X_tr.reshape(-1, batch_size, rows, columns, channels)

    y_tr = y_tr.reshape(-1, batch_size, channels)

    random_ind = np.random.choice(
        [i for i in range(X_ts.shape[0])],
        sample,
        replace=False
    )
    X_ts = X_ts.reshape(-1, rows, columns, channels)

    X_ts = X_ts[random_ind]
    y_ts = y_ts[random_ind]

    return X_tr, y_tr, X_ts, y_ts


def get_representations(X_tr, y_tr, X_ts, conv_1_enc, conv_2_enc, dense_enc):
    """Train model and get batch-tobatch generated representation
    """
    representations = {
        'conv_1': {},
        'conv_2': {},
        'dense': {},
    }
    representations['conv_1'][0] = conv_1_enc.predict(X_ts)
    representations['conv_2'][0] = conv_2_enc.predict(X_ts)
    representations['dense'][0] = dense_enc.predict(X_ts)

    index_batch = 1
    rnd_batches = [batch for batch in range(X_tr.shape[0])]
    random.shuffle(rnd_batches)
    for rnd_batch in tqdm(rnd_batches):

        model.train_on_batch(
            X_tr[rnd_batch, :, :, :, :],
            y_tr[rnd_batch, :, :]
        )
        representations['conv_1'][index_batch] = conv_1_enc.predict(X_ts)
        representations['conv_2'][index_batch] = conv_2_enc.predict(X_ts)
        representations['dense'][index_batch] = dense_enc.predict(X_ts)

        index_batch += 1

    return representations


###############################################################################


X_tr, y_tr, X_ts, y_ts = load_data()

selected_samples = []
for unique_class in range(10):

    index = np.argwhere(y_ts.flatten() == unique_class).flatten()[0]
    selected_samples.append(index)

np.save('results//images//fashion_mnist.npy', X_ts[selected_samples])

model, conv_1_enc, conv_2_enc, dense_enc = create_model_encoders(
    X_tr[0],
    y_tr[0]
)

representations = get_representations(
    X_tr=X_tr,
    y_tr=y_tr,
    X_ts=X_ts,
    conv_1_enc=conv_1_enc,
    conv_2_enc=conv_2_enc,
    dense_enc=dense_enc
)

for batch_n in range(len(representations['conv_1'])):

    representations['conv_1'][batch_n] =   \
        representations['conv_1'][batch_n][selected_samples]
    representations['conv_2'][batch_n] =    \
        representations['conv_2'][batch_n][selected_samples]

with open('results//filters//fashion_mnist_conv_1.pkl', 'wb') as out_conv_1:
    pickle.dump(representations['conv_1'], out_conv_1, pickle.HIGHEST_PROTOCOL)
with open('results//filters//fashion_mnist_conv_2.pkl', 'wb') as out_conv_2:
    pickle.dump(representations['conv_2'], out_conv_2, pickle.HIGHEST_PROTOCOL)

embeddings, relations = prepare_alligned_umap_data(
    representations['dense'],
    max_time_index=len(representations['dense'])
)

mapper = AlignedUMAP(
    metric='cosine',
    n_neighbors=30,
    alignment_regularisation=0.1,
    alignment_window_size=10,
    n_epochs=200,
    random_state=42,
    verbose=True
)
mapper.fit(embeddings, relations=relations)
reductions = mapper.embeddings_

n_embeddings = len(reductions)
embedding_df = pd.DataFrame(
    np.vstack(reductions),
    columns=('UMAP_1', 'UMAP_2')
)
embedding_df['batcn_n'] = np.repeat(
    np.array([i for i in range(n_embeddings)]),
    reductions[0].shape[0]
)
embedding_df['id'] = np.tile(
    np.arange(reductions[0].shape[0]),
    n_embeddings
)
embedding_df['class'] = np.tile(
    y_ts.flatten(),
    n_embeddings
)

###############################################################################

embedding_df.to_csv(
    'results//embeddings//fashion_mnist.csv'
)
