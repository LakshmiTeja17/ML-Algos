from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from matplotlib.patches import Patch


# import copy

dataset = pd.read_csv('data.csv')

target = np.array(dataset['category'])
del dataset['category']
data = np.array(dataset)
t = np.empty(len(target), dtype= int)

t[target == 'Full democracy'] = 0
t[target == 'Flawed democracy'] = 1
t[target == 'Hybrid regime'] = 2
t[target == 'Authoritarian'] = 3

data = data - np.mean(data, axis=0)

size = 15


# Initialization and training
som = MiniSom(15, 15, len(dataset.columns), sigma=1.5, learning_rate=.7, activation_distance='euclidean',
              topology='hexagonal', neighborhood_function='gaussian', random_seed=10)

som.pca_weights_init(data)
som.train_batch(data, 1000, verbose=True)

from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

f = plt.figure(figsize=(10,10))
ax = f.add_subplot(111)

ax.set_aspect('equal')

xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()

for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):

        wy = yy[(i, j)]*2/np.sqrt(3)*3/4

        hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                      facecolor=cm.Blues(umatrix[i, j]), alpha=.4, edgecolor='gray')
        ax.add_patch(hex)

# print(data)
markers = ['o', '+', 'x', '*']
colors = ['C0', 'C1', 'C2', 'C3']
for cnt, x in enumerate(list(data)):
    w = som.winner(x)  # getting the winner
    # palce a marker on the winning position for the sample xx
    wx, wy = som.convert_map_to_euclidean(w)
    wy = wy*2/np.sqrt(3)*3/4
    plt.plot(wx, wy, markers[int(t[cnt])], markerfacecolor='None',
             markeredgecolor=colors[int(t[cnt])], markersize=12, markeredgewidth=2)

xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange-.5, xrange)
plt.yticks(yrange*2/np.sqrt(3)*3/4, yrange)

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,
                            orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

legend_elements = [Line2D([0], [0], marker='o', color='C0', label='full democracy',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='+', color='C1', label='flawed democracy',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='x', color='C2', label='hybrid regime',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='*', color='C3', label='authoritarian',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   ]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left',
          borderaxespad=0., ncol=3, fontsize=14)

plt.show()

feature_names = dataset.columns
W = som.get_weights()
plt.figure(figsize=(10, 10))
plt.title("Feature Maps")
for i, f in enumerate(feature_names):
    plt.subplot(3, 3, i+1)
    plt.title(f)
    plt.pcolor(W[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(size+1))
    plt.yticks(np.arange(size+1))
plt.tight_layout()
plt.savefig("Feature Maps")
plt.show()

Z = np.zeros((size, size))
plt.figure(figsize=(8, 8))
for i in np.arange(som._weights.shape[0]):
    for j in np.arange(som._weights.shape[1]):
        feature = np.argmax(W[i, j , :])
        plt.plot([j+.5], [i+.5], 'o', color='C'+str(feature),
                 marker='s', markersize=24)

legend_elements = [Patch(facecolor='C'+str(i),
                         edgecolor='w',
                         label=f) for i, f in enumerate(feature_names)]

plt.legend(handles=legend_elements,
           loc='center left',
           bbox_to_anchor=(1, .95))

plt.xlim([0, size])
plt.ylim([0, size])
plt.title("Most Relevant Feature Plane")
plt.savefig("Most Relevant Feature Plane")
plt.show()