import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.io import loadmat

from sklearn.decomposition import PCA

from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe

def plot_trajectory(ax, x, y,
                    color="black",
                    outline="black",
                    circle=True,
                    arrow=True,
                    circle_size=0.05,
                    arrow_size=0.05):
    """
    Plot a single neural trajectory in a 2D plane.
    
    Args
    ----
        ax: Axis used for plotting.
        
        x: Values of variable on x-axis.
        
        y: Values of variable on y-axis.
        
        color: Fill color of line to be plotted. Defaults to "black".
        
        outline: Outline color of line. Defaults to "black".
        
        circle: True if the trajectory should have a circle at its starting state.
        
        arrow: True if the trajectory should have an arrow at its ending state.
        
    """
    ax.plot(x, y,
             color=color,
             path_effects=[pe.Stroke(linewidth=2, foreground=outline), pe.Normal()])
        
    if circle:
        circ = plt.Circle((x[0], y[0]),
                          radius=circle_size, 
                          facecolor=color,
                          edgecolor="black")
        ax.add_artist(circ)

    if arrow:
        dx = x[-1] - x[-2]
        dy = y[-1] - y[-2]
        px, py = (x[-1], y[-1])
        ax.arrow(px, py , dx, dy, 
                  facecolor=color, 
                  edgecolor=outline,
                  length_includes_head=True,
                  head_width=arrow_size)

def plot_projections(data_list,
                     x_idx=0,
                     y_idx=1,
                     axis=None,
                     arrows=True,
                     circles=True,
                     arrow_size=0.05,
                     circle_size=0.05):
    """
    Plot trajectories found via jPCA or PCA. 
    
    Args
    ----
        data_list: List of trajectories, where each entry of data_list is an array of size T x D, 
                   where T is the number of time-steps and D is the dimension of the projection.
        x_idx: column of data which will be plotted on x axis. Default 0.
        y_idx: column of data which will be plotted on y axis. Default 0.
        arrows: True to add arrows to the trajectory plot.
        circles: True to add circles at the beginning of each trajectory.
        sort_colors: True to color trajectories based on the starting x coordinate. This mimics
                     the jPCA matlab toolbox.
    """
    if axis is None:
        fig = plt.figure(figsize=(5,5))
        axis = fig.add_axes([1, 1, 1, 1])

    colormap = plt.cm.RdBu
    colors = np.array([colormap(i) for i in np.linspace(0, 1, len(data_list))])
    data_list = [data[:, [x_idx, y_idx]] for data in data_list]

    start_x_list = [data[0,0] for data in data_list]
    color_indices = np.argsort(start_x_list)

    for i, data in enumerate(np.array(data_list)[color_indices]):
        plot_trajectory(axis,
                        data[:, 0],
                        data[:, 1],
                        color=colors[i],
                        circle=circles,
                        arrow=arrows,
                        arrow_size=arrow_size,
                        circle_size=circle_size)


def load_churchland_data(path):
    """
    Load data from Churchland, Cunningham et al, Nature 2012
    Data available here: 
    https://churchland.zuckermaninstitute.columbia.edu/content/code
    
    Returns a 3D data array, C x T x N. T is the number of time bins,
    and N is the number of neurons (218), and C is the number of conditions.

    Note: Loading a .mat struct in Python is quite messy beause the formatting is messed up,
    which is why we need to do some funky indexing here.
    """
    struct = loadmat(path)
    conditions = struct["Data"][0]

    # For this dataset, times are the same for all conditions,
    # but they are formatted strangely -- each element of the times
    # vector is in a separate list.
    datas = None
    times = [t[0] for t in conditions[0][1]]
    
    for cond in conditions:
        spikes = cond[0]
        if datas is None:
            datas = spikes
        else:
            datas = np.dstack((datas, spikes))
    
    datas = np.moveaxis(datas, 2, 0)
    datas = [x for x in datas]
    return datas, times

def ensure_datas_is_list(f):
    def wrapper(self, datas, **kwargs):
        datas = [datas] if not isinstance(datas, list) else datas
        return f(self, datas, **kwargs)
    return wrapper

