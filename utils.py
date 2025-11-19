import matplotlib.pyplot as plt

def print_header(text: str):
    print('*'*50)
    print(text)
    print('*'*50)


def plot_comparison(
        losses: list,
        title: str,
        legend: list,
        fn: str,
):
    for loss in losses:
        plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend(legend)
    plt.savefig(fn)
    plt.close()