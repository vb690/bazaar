import numpy as np

from umap import AlignedUMAP

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow_addons.optimizers import CyclicalLearningRate

import matplotlib

import seaborn as sns

from utilities.models_utils import build_MLP, get_weights
from utilities.viz_utils import save_3D_animation, save_2D_animation


# det the aesthetics for the plot
def sns_styleset():
    """Set the global rcParams for matplolib.
    """
    sns.set(context='paper', style='whitegrid', font='DejaVu Sans')
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['axes.linewidth'] = 1
    matplotlib.rcParams['xtick.major.width'] = 1
    matplotlib.rcParams['ytick.major.width'] = 1
    matplotlib.rcParams['xtick.major.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 3
    matplotlib.rcParams['xtick.minor.size'] = 2
    matplotlib.rcParams['ytick.minor.size'] = 2
    matplotlib.rcParams['font.size'] = 13
    matplotlib.rcParams['axes.titlesize'] = 13
    matplotlib.rcParams['axes.labelsize'] = 13
    matplotlib.rcParams['legend.fontsize'] = 13
    matplotlib.rcParams['xtick.labelsize'] = 13
    matplotlib.rcParams['ytick.labelsize'] = 13


sns_styleset()

# ########################### DEFINE VARIABLES ################################

(X_tr, y_tr), (X_ts, y_ts) = fashion_mnist.load_data()
X_tr = X_tr.reshape(-1, 28*28)
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_ts = X_ts.reshape(-1, 28*28)
X_ts = scaler.transform(X_ts)

epochs = 2
batch_size = 250

cyclical_learning_rate = CyclicalLearningRate(
    initial_learning_rate=3e-4,
    maximal_learning_rate=3e-2,
    step_size=(X_tr.shape[0] // batch_size) * 2,
    scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    scale_mode='cycle'
)

target_optimizers = {
    'SGD': 'SGD',
    'SGD(lr=1)': SGD(lr=1.),
    'SGD(lr=1e-6)': SGD(lr=.000001),
    'Adam': 'adam',
    'Adam CLR': Adam(learning_rate=cyclical_learning_rate),
    'FTRL': 'Ftrl'
}

# ########################### MODEL TRAINING ##################################

total_weights = []
total_train_losses = []
total_test_losses = []

for opt_name, optimizer in target_optimizers.items():

    print('')
    print(f'Training with Optimizer: {opt_name}')
    print('')

    model = build_MLP(X_tr, y_tr)
    weights_dict, train_losses, test_losses, steps = get_weights(
        model=model,
        X=(X_tr, X_ts),
        y=(y_tr, y_ts),
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        loss='sparse_categorical_crossentropy'
    )

    weights = [np.array(weights_dict[weight_idx]) for weight_idx in [0, 1, 2]]
    total_weights.append(weights)
    total_train_losses.append(train_losses)
    total_test_losses.append(test_losses)

# ################################ EXTRACT EMBEDDINGS #########################

to_embed_weights = []
for layer in range(3):

    stacked_w = np.vstack(
        [total_weights[optimizer][layer] for optimizer
            in range(len(target_optimizers))]
    )
    stacked_w = StandardScaler().fit_transform(stacked_w)
    to_embed_weights.append(
        stacked_w
    )

rela = [{
    key: key for key in range(to_embed_weights[0].shape[0])} for i in range(2)
]
mapper = AlignedUMAP(
    n_components=2,
    metric='manhattan',
    n_neighbors=30
).fit(to_embed_weights, relations=rela)

emb_space_sizes = []
for emb in mapper.embeddings_:

    emb_space_sizes.append(
        [
            np.append(
                emb.min(0), np.array(total_train_losses).flatten().min()
            ),
            np.append(
                emb.max(0), np.array(total_train_losses).flatten().max()
            )
        ]
    )


# ################################ SAVE ANIMATIONS ############################

for index, opt_name in enumerate(target_optimizers.keys()):

    emb_size = len(total_test_losses[index])

    start = emb_size * index
    stop = start + emb_size

    save_3D_animation(
        [emb[start:stop] for emb in mapper.embeddings_],
        emb_space_sizes=emb_space_sizes,
        train_losses=total_train_losses[index],
        test_losses=total_test_losses[index],
        opt_name=opt_name,
        horizon_size=50,
        n_bins=20,
        s=20
    )

save_2D_animation(
    embeddings=mapper.embeddings_,
    target_optimizers=[
        'SGD',
        'SGD(lr=1)',
        'SGD(lr=1e-6)',
        'Adam',
        'Adam CLR',
        'FTRL'
    ],
    emb_space_sizes=emb_space_sizes,
    total_train_losses=total_train_losses,
    total_test_losses=total_test_losses,
    n_bins=100,
    cmap_name='jet'
)
