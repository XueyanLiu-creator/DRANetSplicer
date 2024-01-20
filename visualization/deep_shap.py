import shap
import numpy as np
import visualization.vis_seq as vis_seq

def deep_shap(x, background_data, model, title):
    # ensure the shape of x is (batch_size, seq_len, 4, 1)
    if len(x.shape)==4 and x.shape[3]==1:
        assert True
    elif len(x.shape)==3 and x.shape[2]==4:
        x = np.expand_dims(x, axis=-1)
    else:
        assert False, "The shape of x is not (batch_size, seq_len, 4, 1) or (batch_size, seq_len, 4)"

    # ensure the shape of background_data is (batch_size, seq_len, 4, 1)
    if len(background_data.shape)==4 and background_data.shape[3]==1:
        assert True
    elif len(background_data.shape)==3 and background_data.shape[2]==4:
        background_data = np.expand_dims(background_data, axis=-1)
    else:
        assert False, "The shape of background_data is not (batch_size, seq_len, 4, 1) or (batch_size, seq_len, 4)"

    # solving runtime errors
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3" ] = shap.explainers._deep.deep_tf.passthrough

    # explain predictions of the model
    explainer = shap.DeepExplainer(model=(model.input, model.layers[-1].output), data=background_data)
    shap_values = explainer.shap_values(x)

    # reshape (batch_size, seq_len, 4, 1)->(batch_size, seq_len, 4)
    new_shape = (x.shape[0], x.shape[1], x.shape[2])
    x = np.reshape(x, new_shape)

    # model contribution score to positive classification
    task0 = np.reshape(shap_values[0], new_shape)
    # model contribution score to negative classification
    task1 = np.reshape(shap_values[1], new_shape)

    # calculate the average weighted contribution fraction per nucleotide at each position
    def get_weights(scores, original_onehot_input):
        scores = np.sum(scores, axis=2)
        contribution_scores = []
        for idx in range(0, scores.shape[0]):
            scores_for_idx = scores[idx]
            original_onehot = original_onehot_input[idx]
            scores_for_idx = original_onehot*scores_for_idx[:,None]
            contribution_scores.append(scores_for_idx)
        contribution_scores = np.array(contribution_scores,dtype=float)
        weighted_contribution_scores = 100*contribution_scores.shape[0]*contribution_scores/np.sum(np.absolute(contribution_scores), axis=(0,1))
        average_weighted_contribution_scores = np.sum(weighted_contribution_scores, axis=0)/weighted_contribution_scores.shape[0]
        return average_weighted_contribution_scores

    ave_wcs_positive = get_weights(task0, x)
    ave_wcs_negative = get_weights(task1, x)

    lower = 1
    upper = ave_wcs_positive.shape[0]

    fig_positive = vis_seq.plot_weights(ave_wcs_positive, lower, upper, title=f'{title} positive', ylabel='average wcs', figsize=(20,3), length_padding=1, subticks_frequency=50)
    fig_negative = vis_seq.plot_weights(ave_wcs_negative, lower, upper, title=f'{title} negative', ylabel='average wcs', figsize=(20,3), length_padding=1, subticks_frequency=50)

    return fig_positive, fig_negative
