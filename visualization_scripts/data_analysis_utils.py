import numpy as np

from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC


def plot_structure(ax, angles=None, rs=None, coords=None, title=None, labels=None, fontsize=9, rays=True, s=None,
                   color=None, colors=None):
    # make self-direction top
    if coords is None:
        theor_complex_rot = rs * np.exp(1j * (np.array(angles) + np.pi / 2))
        coords = np.column_stack([theor_complex_rot.real, theor_complex_rot.imag])

    if color:
        colors = color

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()

    ax.scatter(x=coords[:, 0], y=coords[:, 1], color=colors, s=s)

    # rays
    if rays:
        for dot in coords:
            ax.plot([0, dot[0]], [0, dot[1]], color="gray", linewidth=1)

    if labels:
        for i, v in enumerate(labels):
            ax.annotate(v, coords[i], fontsize=fontsize)

    # if title.startswith("Theo"):
    #     title = "Theoretical structure"
    # else:
    #     title = "GPT-4o-0513 structure"

    ax.set_title(title, y=1.05, fontsize=15)

    # plt.savefig(f"{title}.svg")
    # plt.savefig(f"./svgs/{title}.svg")



def classify_dots(data, labels):

    # clf = OneVsOneClassifier(LinearRegression()).fit(data, labels)
    clf = OneVsOneClassifier(LinearSVC(dual="auto")).fit(data, labels)
    pred = clf.predict(data)
    acc = np.mean(pred == labels)
    return acc
