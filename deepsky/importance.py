import numpy as np
import pandas as pd


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
