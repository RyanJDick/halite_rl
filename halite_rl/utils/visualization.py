
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pandas as pd
import seaborn as sn

def plot_confusion_matrix(cm, labels):
    df_cm = pd.DataFrame(
        cm,
        index=[l + "_gt" for l in labels],
        columns=[l + "_pred" for l in labels],
    )
    plt.figure(figsize=(15,12))
    sn.heatmap(df_cm, annot=True)
    #plt.savefig("confusion_matrix.png")
    cm_img = save_figure_to_np_array()
    plt.close() # Figure must be explicitly closed.
    return cm_img

def save_figure_to_np_array():
    fig = plt.gcf()
    w, h = fig.get_size_inches() * fig.get_dpi()
    w = int(w)
    h = int(h)
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw() # draw the canvas, cache the renderer.
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    return img.reshape(h, w, 3)
