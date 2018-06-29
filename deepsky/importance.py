import numpy as np
import pandas as pd
import keras.backend as K

def variable_importance(data, labels, variable_names, model_name, model, score_func, permutations=30,
                        sklearn_model=False,
                        mean_model=False):
    if sklearn_model:
        preds = model.predict_proba(data)[:, 1]
    else:
        preds = model.predict(data)[:, 0]
    score = score_func(labels, preds)
    indices = np.arange(preds.shape[0])
    perm_data = np.copy(data)
    var_scores = pd.DataFrame(index=np.arange(permutations + 1), columns=variable_names, dtype=float)
    var_scores.loc[0, :] = score
    data_shape_len = len(data.shape)
    print(data.shape)
    for v, variable in enumerate(variable_names):
        for p in range(1, permutations + 1):
            np.random.shuffle(indices)
            if mean_model and data_shape_len == 2:
                perm_data[:, v] = data[indices, v]
            elif mean_model and data_shape_len == 3:
                perm_data[:, :, v] = data[indices, :, v]
            else:
                perm_data[:, :, :, v] = data[indices, :, :, v]
            if sklearn_model:
                perm_preds = model.predict_proba(perm_data)[:, 1]
            else:
                perm_preds = model.predict(perm_data)[:, 0]
            var_scores.loc[p, variable] = score_func(labels, perm_preds)
        if mean_model and not data_shape_len == 2:
            perm_data[:, v] = data[:, v]
        elif mean_model and data_shape_len == 3:
            perm_data[:, :, v] = data[:, :, v]
        else:
            perm_data[:, :, :, v] = data[:, :, :, v]
        score_diff = (var_scores.loc[0, variable] - var_scores.loc[1:, variable]) /  var_scores.loc[0, variable]
        print(model_name, variable, score_diff.mean(), score_diff.std())
    return var_scores


def variable_importance_faster(data, labels, variable_names, model_name, model, score_funcs, permutations=30,
                        sklearn_model=False,
                        mean_model=False):
    if sklearn_model:
        preds = model.predict_proba(data)[:, 1]
    else:
        preds = model.predict(data)[:, 0]
    scores = [sf(labels, preds) for sf in score_funcs]
    indices = np.arange(preds.shape[0])
    perm_data = np.copy(data)
    var_scores = []
    for s in range(len(score_funcs)):
        var_scores.append(pd.DataFrame(index=np.arange(permutations + 1), columns=variable_names, dtype=float))
        var_scores[-1].loc[0, :] = scores[s]
    data_shape_len = len(data.shape)
    for p in range(1, permutations + 1):
        np.random.shuffle(indices)
        for v, variable in enumerate(variable_names):
            if mean_model and data_shape_len == 2:
                perm_data[:, v] = data[indices, v]
            elif mean_model and data_shape_len == 3:
                perm_data[:, :, v] = data[indices, :, v]
            else:
                perm_data[:, :, :, v] = data[indices, :, :, v]
            if sklearn_model:
                perm_preds = model.predict_proba(perm_data)[:, 1]
            else:
                perm_preds = model.predict(perm_data)[:, 0]
            for s, score_func in enumerate(score_funcs):
                var_scores[s].loc[p, variable] = score_func(labels, perm_preds)
            if mean_model and not data_shape_len == 2:
                perm_data[:, v] = data[:, v]
            elif mean_model and data_shape_len == 3:
                perm_data[:, :, v] = data[:, :, v]
            else:
                perm_data[:, :, :, v] = data[:, :, :, v]
            print(model_name, variable)
    return var_scores


def activated_analogs(norm_data, cnn_model, num_analogs=16, filter_index=(0, 0), dense_layer_index=-2,
                      conv_layer_index=-6):
    """
    For a given convolutional neural network, identify the examples that most activate a given set of
    neurons in a convolutional layer.

    Args:
        norm_data: Normalized input data for the convolutional neural network
        cnn_model: Keras convolutional neural network model object
        num_analogs: Number of activated input examples to store for each neuron
        filter_index: Spatial array index of convolution filters being evaluated
        dense_layer_index: Index of the final dense layer that connects the convolutions and outputs
        conv_layer_index: Index of the activation of the final convolutional layer.

    Returns:
        dense_weights (array of weight values), top_analog_ids (input data indices associated with each neuron),
        top_analog_activations (the magnitude of the activation), top_analog_gradients (gradients with respect
        to the input for each of the top analogs)
    """
    dense_weights = pd.Series(cnn_model.layers[dense_layer_index].get_weights().reshape(
        cnn_model.layers[conv_layer_index].output_shape[1:])[filter_index], name="Weights")
    top_analog_ids = pd.DataFrame(np.zeros((dense_weights.shape[-1], num_analogs), dtype=int),
                                  columns=["Analog_ID_{0:02d}".format(a) for a in range(num_analogs)])
    top_analog_activations = pd.DataFrame(np.zeros((dense_weights.shape[-1], num_analogs)),
                                          columns=["Analog_Act_{0:02d}".format(a) for a in range(num_analogs)])
    top_analog_gradients = np.zeros([dense_weights.shape[-1], num_analogs] + list(norm_data.shape[1:]))
    for w in range(dense_weights.shape[-1]):
        filter_out = cnn_model.layers[conv_layer_index].output[:, filter_index[0], filter_index[1], w]
        act_func = K.function([cnn_model.input, K.learning_phase()],
                              [filter_out])
        loss = (filter_out - 4) ** 2
        grad = K.gradients(loss, cnn_model.input)[0]
        grad /= K.maximum(K.std(grad), K.epsilon())
        grad_func = K.function([cnn_model.input, K.learning_phase()], [grad])
        max_acts = act_func([norm_data, 0])[0]
        top_analog_ids.loc[w] = np.argsort(max_acts)[::-1][:num_analogs]
        top_analog_activations.loc[w] = max_acts[top_analog_ids[w]]
        top_analog_gradients[w] = grad_func([norm_data[top_analog_ids], 0])[0]
    combined_info = pd.concat([top_analog_ids, top_analog_activations, dense_weights], axis=1)
    return combined_info, top_analog_gradients