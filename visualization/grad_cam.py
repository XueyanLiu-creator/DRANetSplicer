import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
import visualization.vis_seq as vis_seq

def grad_cam(model, x, category_index, layer_name):

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer_out = grad_model(x)
        class_out = model_out[:, category_index]
        grads = tape.gradient(class_out, last_conv_layer_out)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_out), axis=-1)
    if heatmap[0][0] < 0.0:
        heatmap = heatmap*-1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[2]))

    return heatmap

def grad_cam_heatmap(x_set, y_set, model, layer_name='conv2d_20', title=''):
    
    model = model
    seq_set= x_set
    seq_label = y_set[:,1]
    layer_name = layer_name

    all_matrix_ss = []
    all_matrix_no_ss = []

    for i in tqdm(range(x_set.shape[0])):
        i = int(i)
        x = seq_set[i]
        x = x.reshape(-1, 402, 4)

        category_index = int(seq_label[i])

        heatmap = grad_cam(model, x, category_index, layer_name)

        if int(category_index) == 1:
            all_matrix_no_ss.append(heatmap)
        elif int(category_index) == 0:
            all_matrix_ss.append(heatmap)

    all_matrix_no_ss = np.array(all_matrix_no_ss)
    all_matrix_ss = np.array(all_matrix_ss)
    ave_matrix_no_ss = np.mean(np.array(all_matrix_no_ss), axis=0)
    ave_matrix_ss = np.mean(np.array(all_matrix_ss), axis=0)

    fig_positive = vis_seq.plot_map(ave_matrix_ss, title=f'{title} positive', figsize=(12,1))
    fig_negative = vis_seq.plot_map(ave_matrix_no_ss, title=f'{title} negative', figsize=(12,1))

    return fig_positive, fig_negative
