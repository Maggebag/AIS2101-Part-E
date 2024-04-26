import matplotlib.pyplot as plt
import seaborn as sns


def plot_all(dataset):
    plot_histograms(dataset)
    plot_box_plots(dataset)
    plot_bar_plot(dataset)
    plot_pair_plot(dataset)
    plot_correlation_heatmap(dataset)


def plot_histograms(dataset):
    dataset.hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()


def plot_box_plots(dataset):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 16))
    axes = axes.flatten()

    for i, column in enumerate(dataset.columns[:-1]):  # Exclude the last column (Class)
        sns.boxplot(x='Class', y=column, data=dataset, ax=axes[i])
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel(column)
        axes[i].set_title(f'Box plot of {column} by Class')

    plt.tight_layout()
    plt.show()


def plot_bar_plot(dataset):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=dataset)
    plt.xticks(rotation=45)
    plt.show()


def plot_pair_plot(dataset):
    sns.pairplot(dataset, hue='Class')
    plt.show()


def plot_correlation_heatmap(dataset):
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.drop(columns=['Class']).corr(), annot=True, cmap='coolwarm')
    plt.show()

