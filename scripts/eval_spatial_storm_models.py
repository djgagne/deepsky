import numpy as np
import pandas as pd
import xarray as xr
from keras.models import save_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from deepsky.gan import normalize_multivariate_data
from deepsky.metrics import brier_score, brier_skill_score
from deepsky.models import hail_conv_net, LogisticPCA
import pickle
import itertools as it
from glob import glob
from os.path import join, exists
from os import mkdir
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    if not exists(config["out_path"]):
        mkdir(config["out_path"])
    all_param_combos = {}
    for model_name in config["model_names"]:
        param_names = sorted(list(config[model_name].keys()))
        all_param_combos[model_name] = pd.DataFrame(list(it.product(*[config[model_name][conv_name]
                                                    for conv_name in param_names])),
                                                    columns=param_names)
    output_config = config["output"]
    sampling_config = config["sampling"]
    data_path = config["data_path"]
    input_variables = config["input_variables"]
    print("Loading data")
    storm_data, storm_centers, storm_dates, storm_members = load_storm_patch_data(data_path, input_variables)
    storm_norm_data, storm_scaling_values = normalize_multivariate_data(storm_data)
    storm_flat_data = storm_norm_data.reshape(storm_norm_data.shape[0],
                                              storm_norm_data.shape[1] * storm_norm_data.shape[2],
                                              storm_norm_data.shape[3])
    storm_mean_data = storm_flat_data.mean(axis=1)
    output_data, output_centers, output_dates, output_members = load_storm_patch_data(data_path,
                                                                                      [output_config["variable"],
                                                                                       output_config["mask"]])
    max_hail = np.array([output_data[i, :, :, 0][output_data[i, :, :, 1] > 0].max()
                         for i in range(output_data.shape[0])])
    hail_labels = np.where(max_hail >= output_config["threshold"], 1, 0)
    unique_dates = storm_dates.unique()
    evaluate_conv_net(storm_norm_data, storm_centers, storm_dates, storm_members, hail_labels, unique_dates,
                      sampling_config, all_param_combos["conv_net"], config["out_path"])
    evaluate_sklearn_model("logistic_mean", LogisticRegression, storm_mean_data, storm_centers, storm_dates,
                           storm_members, hail_labels, unique_dates, sampling_config,
                           all_param_combos["logistic_mean"], config["out_path"])
    evaluate_sklearn_model("logistic_pca", LogisticPCA, storm_flat_data, storm_centers, storm_dates,
                           storm_members, hail_labels, unique_dates, sampling_config,
                           all_param_combos["logistic_pca"], config["out_path"])
    return


def train_split_generator(values, train_split, num_samples):
    split_index = int(np.round(train_split * values.size))
    for n in range(num_samples):
        shuffled_values = np.random.permutation(values)
        train_values = shuffled_values[:split_index]
        test_values = shuffled_values[split_index:]
        yield train_values, test_values


def evaluate_conv_net(storm_norm_data, storm_centers, storm_dates, storm_members, hail_labels, unique_dates,
                      sampling_config, param_combos, out_path):
    """

    Args:
        storm_norm_data:
        storm_centers:
        storm_dates:
        storm_members:
        hail_labels:
        unique_dates:
        sampling_config:
        param_combos:
        out_path:

    Returns:

    """
    np.random.seed(sampling_config["random_seed"])
    storm_sampler = train_split_generator(unique_dates, sampling_config["train_split"],
                                          sampling_config["num_samples"])
    best_param_combos = []
    sample_scores = pd.DataFrame(index=np.arange(sampling_config["num_samples"]),
                                 columns=["Brier Score", "Brier Score Climo", "Brier Skill Score", "AUC"],
                                 dtype=float)
    for n in range(sampling_config["num_samples"]):
        train_dates, test_dates = next(storm_sampler)
        train_indices = np.where(np.in1d(storm_dates, train_dates))[0]
        test_indices = np.where(np.in1d(storm_dates, test_dates))[0]
        all_members = np.unique(storm_members[train_indices])
        np.random.shuffle(all_members)
        member_split = int(np.round(all_members.size * sampling_config["member_split"]))
        train_members = all_members[:member_split]
        val_members = all_members[:member_split]
        train_member_indices = np.where(np.in1d(storm_members[train_indices], train_members))[0]
        val_member_indices = np.where(np.in1d(storm_members[train_indices], val_members))[0]
        best_config = param_combos.index[0]
        best_score = -1.0
        param_scores = pd.DataFrame(index=np.arange(param_combos.shape[0]), columns=["Brier Skill Score", "AUC"])
        for c in param_combos.index:
            print(param_combos.loc[c])
            hail_conv_net_model = hail_conv_net(**param_combos.loc[c].to_dict())
            hail_conv_net_model.fit(storm_norm_data[train_indices][train_member_indices],
                                    hail_labels[train_indices][train_member_indices],
                                    batch_size=param_combos.loc[c, "batch_size"],
                                    epochs=param_combos.loc[c, "num_epochs"], verbose=2)
            val_preds = hail_conv_net_model.predict(storm_norm_data[train_indices][val_member_indices]).ravel()
            param_scores.loc[c, "Brier Skill Score"] = brier_skill_score(hail_labels[train_indices][val_member_indices],
                                                                         val_preds)
            param_scores.loc[c, "AUC"] = roc_auc_score(hail_labels[train_indices][val_member_indices],
                                                                         val_preds)
            if param_scores[c] > best_score:
                best_config = c
                best_score = param_scores[c]
            del hail_conv_net_model
        param_scores.to_csv(join(out_path, "conv_net_param_scores_sample_{0:03d}.h5".format(n)),
                            index_label="Param Combo")
        best_param_combos.append(best_config)
        print("Best Config")
        print(param_combos.loc[best_config])
        print("Train Conv Net")
        hail_conv_net_model = hail_conv_net(**param_combos.loc[best_config].to_dict())
        hail_conv_net_model.fit(storm_norm_data[train_indices],
                                hail_labels[train_indices],
                                batch_size=param_combos.loc[best_config, "batch_size"],
                                epochs=param_combos.loc[best_config, "num_epochs"], verbose=2)
        print("Scoring Conv Net")
        test_preds = hail_conv_net_model.predict(storm_norm_data[test_indices]).ravel()
        test_pred_frame = pd.DataFrame({"indices": test_indices,
                                        "lon": storm_centers[test_indices, 0],
                                        "lat": storm_centers[test_indices, 1],
                                        "dates": storm_dates[test_indices],
                                        "members": storm_members[test_indices],
                                        "conv_net": test_preds,
                                        "label": hail_labels[test_indices]},
                                       columns=["indices", "lon", "lat", "dates", "members", "conv_net", "label"])
        test_pred_frame.to_csv(join(out_path, "predictions_conv_net_sample_{0:03d}.h5".format(n)), index=False)
        sample_scores.loc[n, "Brier Score"] = brier_score(hail_labels[test_indices], test_preds)
        sample_scores.loc[n, "Brier Score Climo"] = brier_score(hail_labels[test_indices],
                                                                hail_labels[test_indices].mean())
        sample_scores.loc[n, "Brier Skill Score"] = brier_skill_score(hail_labels[test_indices], test_preds)
        sample_scores.loc[n, "AUC"] = roc_auc_score(hail_labels[test_indices], test_preds)
        save_model(hail_conv_net_model, join(out_path, "hail_conv_net_sample_{0:03d}.h5".format(n)))
        del hail_conv_net_model
    sample_scores.to_csv(join(out_path, "conv_net_sample_scores.csv"), index_label="Sample")
    best_config_frame = param_combos.loc[best_param_combos]
    best_config_frame = best_config_frame.reset_index()
    best_config_frame.to_csv(join(out_path, "conv_net_best_params.csv"), index_label="Sample")
    return


def evaluate_sklearn_model(model_name, model_obj, storm_data, storm_centers, storm_dates, storm_members, hail_labels,
                           unique_dates, sampling_config, param_combos, out_path):
    np.random.seed(sampling_config["random_seed"])
    storm_sampler = train_split_generator(unique_dates, sampling_config["train_split"],
                                          sampling_config["num_samples"])
    best_param_combos = []
    sample_scores = pd.DataFrame(index=np.arange(sampling_config["num_samples"]),
                                 columns=["Brier Score", "Brier Score Climo", "Brier Skill Score", "AUC"],
                                 dtype=float)
    for n in range(sampling_config["num_samples"]):
        train_dates, test_dates = next(storm_sampler)
        train_indices = np.where(np.in1d(storm_dates, train_dates))[0]
        test_indices = np.where(np.in1d(storm_dates, test_dates))[0]
        all_members = np.unique(storm_members[train_indices])
        np.random.shuffle(all_members)
        member_split = int(np.round(all_members.size * sampling_config["member_split"]))
        train_members = all_members[:member_split]
        val_members = all_members[:member_split]
        train_member_indices = np.where(np.in1d(storm_members[train_indices], train_members))[0]
        val_member_indices = np.where(np.in1d(storm_members[train_indices], val_members))[0]
        best_config = param_combos.index[0]
        best_score = -1.0
        param_scores = pd.DataFrame(index=np.arange(param_combos.shape[0]), columns=["Brier Skill Score", "AUC"])
        for c in param_combos.index:
            print(param_combos.loc[c])
            model_inst = model_obj(**param_combos.loc[c].to_dict())
            model_inst.fit(storm_data[train_indices][train_member_indices],
                           hail_labels[train_indices][train_member_indices])
            val_preds = model_inst.predict_proba(storm_data[train_indices][val_member_indices])[:, 1]
            param_scores.loc[c, "Brier Skill Score"] = brier_skill_score(hail_labels[train_indices][val_member_indices],
                                                                         val_preds)
            param_scores.loc[c, "AUC"] = roc_auc_score(hail_labels[train_indices][val_member_indices],
                                                       val_preds)
            if param_scores[c] > best_score:
                best_config = c
                best_score = param_scores[c]
            del model_inst
        param_scores.to_csv(join(out_path, "{0}_param_scores_sample_{1:03d}.h5".format(model_name, n)),
                            index_label="Param Combo")
        best_param_combos.append(best_config)
        print("Best Config")
        print(param_combos.loc[best_config])
        print("Train Best " + model_name)
        model_inst = model_obj(**param_combos.loc[best_config].to_dict())
        model_inst.fit(storm_data[train_indices],
                       hail_labels[train_indices])
        print("Scoring " + model_name)
        test_preds = model_inst.predict_proba(storm_data[test_indices])[:, 1]
        test_pred_frame = pd.DataFrame({"indices": test_indices,
                                        "lon": storm_centers[test_indices, 0],
                                        "lat": storm_centers[test_indices, 1],
                                        "dates": storm_dates[test_indices],
                                        "members": storm_members[test_indices],
                                        model_name: test_preds,
                                        "label": hail_labels[test_indices]},
                                       columns=["indices", "lon", "lat", "dates", "members", "conv_net", "label"])
        test_pred_frame.to_csv(join(out_path, "predictions_{0}_sample_{1:03d}.h5".format(model_name, n)), index=False)
        sample_scores.loc[n, "Brier Score"] = brier_score(hail_labels[test_indices], test_preds)
        sample_scores.loc[n, "Brier Score Climo"] = brier_score(hail_labels[test_indices],
                                                                hail_labels[test_indices].mean())
        sample_scores.loc[n, "Brier Skill Score"] = brier_skill_score(hail_labels[test_indices], test_preds)
        sample_scores.loc[n, "AUC"] = roc_auc_score(hail_labels[test_indices], test_preds)
        with open(join(out_path, "hail_{0}_sample_{1:03d}.h5".format(model_name, n)), "wb") as model_file:
            pickle.dump(model_inst, model_file, pickle.HIGHEST_PROTOCOL)
        del model_inst
    sample_scores.to_csv(join(out_path, "{0}_sample_scores.csv".format(model_name)), index_label="Sample")
    best_config_frame = param_combos.loc[best_param_combos]
    best_config_frame = best_config_frame.reset_index()
    best_config_frame.to_csv(join(out_path, "{0}_best_params.csv".format(model_name)), index_label="Sample")
    return


def load_storm_patch_data(data_path, variable_names):
    data_patches = []
    centers = []
    valid_dates = []
    members = []
    data_files = sorted(glob(join(data_path, "*.nc")))
    for data_file in data_files:
        member = int(data_file.split("/")[-1][-5:-3])
        ds = xr.open_dataset(data_file)
        patch_arr = []
        all_vars = list(ds.variables.keys())
        if np.all(np.in1d(variable_names, all_vars)):
            centers.append(np.array([ds["longitude"][:, 32, 32], ds["latitude"][:, 32, 32]]).T)
            valid_dates.append(ds["valid_date"].values)
            members.append(np.ones(centers[-1].shape[0], dtype=int) * member)
            for variable in variable_names:
                patch_arr.append(ds[variable][:, 16:-16, 16:-16].values)
            data_patches.append(np.stack(patch_arr, axis=-1)) 
            print(data_file, members[-1].size)
        ds.close()
        del patch_arr
        del ds
    center_arr = np.vstack(centers)
    members_arr = np.concatenate(members)
    valid_date_index = pd.DatetimeIndex(np.concatenate(valid_dates))
    data = np.vstack(data_patches)
    return data, center_arr, valid_date_index, members_arr


if __name__ == "__main__":
    main()
