import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from icecream import ic as ic
from sklearn.manifold import TSNE


def plot_prop_correlation(datasets):
    def get_prop_data(data, prop_idx):
        prop_data = pd.DataFrame(data[prop_idx, :, :])
        return prop_data

    class_names = ['prom', 'non-prom']
    for prop_idx, prop_name in enumerate(datasets.prop_index):

        fig, axes = plt.subplots(3, 2, figsize=(15, 5), sharey=True)
        fig.suptitle(f'Property: {prop_name}')
        for class_idx, dataset in enumerate(datasets.encoded_datasets):
            print(f'{class_idx}')
            prop_data = get_prop_data(dataset, prop_idx)
            dataset_name = class_names[class_idx]

            # # TODO first rows is full index
            # _ax = axes[0, class_idx]
            # plt.Figure()
            # frame_data = prop_data
            # bplot = _ax.boxplot(frame_data)
            # ticks = _ax.get_xticks()
            # lineplot = _ax.plot(ticks, frame_data.mean())
            # # _ax.set_xticklabels(_index)
            # _ax.set_title(f'{dataset_name}')

            for frame in range(3):
                frame_data = prop_data.iloc[:, frame::3]
                _index = prop_data.columns[frame::3]
                _ax = axes[frame, class_idx]
                plt.Figure()
                bplot = _ax.boxplot(frame_data)
                ticks = _ax.get_xticks()
                lineplot = _ax.plot(ticks, frame_data.mean())
                _ax.set_xticklabels(_index)
                _ax.set_title(f'{dataset_name}')

        # for i, _ax in enumerate(axes.flat):
        #     _ax.set(xlabel='x-label', ylabel=f'Frame {i+1}')
        #     _ax.label_outer()
        plt.show()


def plot_tsne(x, y):
    tsne = TSNE(
        n_components=2,
        verbose=1,
        random_state=123,
        perplexity=10,
        learning_rate=10,
        metric='cosine',
        n_jobs=4,
    )
    z = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    plt.Figure()
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    # palette=sns.color_palette("hls", 10),
                    data=df).set(title="MNIST data T-SNE projection")
    plt.show()


def plot_prop_x_prop(class_datasets):
    # plot_prop_correlation(class_datasets)

    prop = 1
    X1 = class_datasets.encoded_datasets[0][prop, :, :]
    y1 = np.zeros(X1.shape[0])

    X2 = class_datasets.encoded_datasets[1][prop, :, :]
    y2 = np.full(X2.shape[0], fill_value=1)

    X3 = np.vstack((X1, X2))
    y3 = np.concatenate((y1, y2))

    ic(X1, X1.shape)
    ic(X2, X2.shape)
    ic(X3, X3.shape, y3)

    X_train = X3
    y_train = y3
    # plot_tsne(X_train, y_train)

    # pos = 59
    # X1 = class_datasets.encoded_datasets[0][:, :, pos]
    # X1 = np.moveaxis(X1, [0, 1], [1, 0])
    # X2 = class_datasets.encoded_datasets[1][:, :, pos]
    # X2 = np.moveaxis(X2, [0, 1], [1, 0])
    # X3 = np.vstack((X1, X2))
    # X_train = X3
    # ic(X3, X3.shape)
    # y_train = y3
    # # plot_tsne(X_train, y_train)

    pos = 60
    propA = 3
    propB = 5
    X1 = class_datasets.encoded_datasets[0][:, :, pos]
    X1 = np.moveaxis(X1, [0, 1], [1, 0])
    X2 = class_datasets.encoded_datasets[1][:, :, pos]
    X2 = np.moveaxis(X2, [0, 1], [1, 0])
    X3 = np.vstack((X1, X2))
    X_train = pd.DataFrame(X3)
    ic(X_train, X_train.shape)
    y_train = y3

    _X1 = pd.DataFrame(X1).groupby([propA, propB]).size().to_frame(name=2).reset_index()
    _X2 = pd.DataFrame(X2).groupby([propA, propB]).size().to_frame(name=2).reset_index()

    fig, axes = plt.subplots(1, 2)
    sns.scatterplot(ax=axes[0], x=_X1.iloc[:, 0], y=_X1.iloc[:, 1], hue=_X1.iloc[:, 2], size=_X1.iloc[:, 2],
                    palette='rocket')
    sns.scatterplot(ax=axes[1], x=_X2.iloc[:, 0], y=_X2.iloc[:, 1], hue=_X2.iloc[:, 2], size=_X2.iloc[:, 2],
                    palette='mako')
    plt.show()