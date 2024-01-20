import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import rcParams

font_path = 'visualization/cmb10.ttf'
font_properties = fm.FontProperties(fname=font_path)
config = {
    "font.family": font_properties.get_name(),
    "font.size": 18,
    "axes.unicode_minus": False
}
rcParams.update(config)

# The following part of the plotting code is from deeplift.viz_sequence

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'#00a961', 1:'#f3331c', 2:'#fab921', 3:'#3766be'}
default_plot_funcs = {0:plot_a, 1:plot_t, 2:plot_g, 3:plot_c}
def plot_weights_given_ax(ax, array, lower, upper, xlabel, ylabel, title,
                 height_padding_factor,
                 length_padding,
                 subticks_frequency,
                 highlight,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(lower-1, upper):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i+0.5, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))

    ax.set_xlim(lower-length_padding, upper+length_padding)
    ax.xaxis.set_ticks(np.arange(lower, upper+1, subticks_frequency))
    ax.axhline(y=0, xmin=0, xmax=402, linestyle='-', linewidth=1, color='gray')
    height_padding_neg = abs(min_neg_height)*(height_padding_factor)
    height_padding_pos = abs(max_pos_height)*(height_padding_factor)
    ylim_min = min_neg_height-height_padding_neg
    ylim_max = max_pos_height+height_padding_pos
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_weights(array, lower=1, upper=402, xlabel='position', ylabel='contribution scores', title='',
                 figsize=(20,2), dpi=150,
                 height_padding_factor=0.1,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    config = {
    "font.family": font_properties.get_name(),
    "font.size": 18,
    "axes.unicode_minus": False
    }
    rcParams.update(config)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.05,bottom=0.20,right=0.98,top=0.85)
    ax = fig.add_subplot(111)
    plot_weights_given_ax(ax=ax, array=array, lower=lower, upper=upper, xlabel=xlabel, ylabel=ylabel, title=title,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)
    return fig

def plot_map(array, title, figsize=(20,6), dpi=100):

    config = {
    "font.family": font_properties.get_name(),
    "font.size": 18,
    "axes.unicode_minus": False
    }
    rcParams.update(config)

    lower = 1
    median = int((array.shape[1]+1)/2)
    upper = array.shape[1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.05,bottom=0.2,right=0.98,top=0.95)
    ax.matshow(array, aspect="auto")
    ax.set_xlim(lower-1, upper)
    ax.xaxis.set_ticks([lower, median, upper])
    ax.set_yticks([])
    ax.tick_params(axis="both", direction="out", which="both", pad=1, labelsize=10, length=1)
    ax.set_title(title, fontfamily=font_properties.get_name(), fontsize=20)

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.5)
    cax = fig.add_axes([0.96, 0.06, 0.015, 0.85])
    cmap = copy.copy(matplotlib.cm.viridis)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    im = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0, 20, 40, 60, 80,100])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8',"1.0"])
    cbar.ax.tick_params(axis="both", direction="out", which="both", pad=1, labelsize=10, length=1)

    return fig