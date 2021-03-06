import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import save_model
from sklearn.metrics import roc_auc_score
from deepsky.gan import normalize_multivariate_data
from deepsky.metrics import brier_score, brier_skill_score
from sklearn.linear_model import LogisticRegression
from deepsky.models import hail_conv_net, LogisticPCA, LogisticGAN, save_logistic_gan
from deepsky.data import load_storm_patch_data
import pickle
import inspect
import itertools as it
from os.path import join, exists
from os import mkdir, environ
import yaml
import argparse
import traceback
from multiprocessing import Pool, Manager


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
        all_param_combos[model_name].to_csv(join(config["out_path"], model_name + "_param_combos.csv"), index_label="Index")
    output_config = config["output"]
    sampling_config = config["sampling"]
    data_path = config["data_path"]
    input_variables = config["input_variables"]
    print("Loading data")
    storm_data, storm_meta = load_storm_patch_data(data_path, input_variables, args.proc)
    storm_norm_data, storm_scaling_values = normalize_multivariate_data(storm_data)
    storm_scaling_values.to_csv(join(config["out_path"], "scaling_values.csv"), index_label="Index")
    storm_flat_data = storm_norm_data.reshape(storm_norm_data.shape[0],
                                              storm_norm_data.shape[1] * storm_norm_data.shape[2],
                                              storm_norm_data.shape[3])
    storm_mean_data = storm_flat_data.mean(axis=1)
    output_data, output_meta = load_storm_patch_data(data_path,
                                                     [output_config["variable"],
                                                      output_config["mask"]], args.proc)
    max_hail = np.array([output_data[i, :, :, 0][output_data[i, :, :, 1] > 0].max()
                         for i in range(output_data.shape[0])])
    max_hail *= 1000
    hail_labels = np.where(max_hail >= output_config["threshold"], 1, 0)
    del output_data
    del output_meta
    del storm_data
    print("Severe hail events: ", np.count_nonzero(hail_labels == 1))
    #evaluate_conv_net(storm_norm_data, storm_meta, hail_labels,
    #                  sampling_config, all_param_combos["conv_net"], config["out_path"])
    #evaluate_sklearn_model("logistic_mean", LogisticRegression, storm_mean_data, storm_meta,
    #                       hail_labels, sampling_config,
    #                       all_param_combos["logistic_mean"], config["out_path"])
    #evaluate_sklearn_model("logistic_pca", LogisticPCA, storm_flat_data, storm_meta,
    #                       hail_labels, sampling_config,
    #                       all_param_combos["logistic_pca"], config["out_path"])
    evaluate_sklearn_model("logistic_gan", LogisticGAN,
                           storm_norm_data, storm_meta,
                           hail_labels, sampling_config,
                           all_param_combos["logistic_gan"], config["out_path"])
    return


def train_split_generator(values, train_split, num_samples):
    split_index = int(np.round(train_split * values.size))
    for n in range(num_samples):
        shuffled_values = np.random.permutation(values)
        train_values = shuffled_values[:split_index]
        test_values = shuffled_values[split_index:]
        yield train_values, test_values


def train_single_conv_net(config_num, device_queue, conv_net_params, out_path):
    device = -1
    try:
        print("Starting process ", config_num)
        device = int(device_queue.get())
        print("Process {0:d} using GPU {1:d}".format(config_num, device))
        environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(device)
        param_scores = {}
        train_data = np.load(join(out_path, "param_train_data.npy"))
        train_labels = np.load(join(out_path, "param_train_labels.npy"))
        val_data = np.load(join(out_path, "param_val_data.npy"))
        val_labels = np.load(join(out_path, "param_val_labels.npy"))
        session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=False,
                                                    gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                    log_device_placement=False))
        K.set_session(session)  
        print("Training ", config_num, device)
        hail_conv_net_model = hail_conv_net(**conv_net_params)
        hail_conv_net_model.fit(train_data,
                                train_labels,
                                batch_size=conv_net_params["batch_size"],
                                epochs=conv_net_params["num_epochs"], verbose=2)
        val_preds = hail_conv_net_model.predict(val_data).ravel()
        param_scores["Brier Skill Score"] = brier_skill_score(val_labels,
                                                            val_preds)
        param_scores["AUC"] = roc_auc_score(val_labels,
                                            val_preds)
        
        print("Scores ", config_num, device, param_scores["Brier Skill Score"], param_scores["AUC"])
        session.close()
        del session
        device_queue.put(device)
        return param_scores, config_num
    except Exception as e:
        if device >= 0:
            device_queue.put(device)
        print(traceback.format_exc())
        raise e


def train_single_sklearn_model(model_name, model_obj, config_num, params, out_path,
                               device_queue=None):
    device = -1
    try:
        print("Starting process ", config_num)
        if device_queue is not None:
            device = int(device_queue.get())
            print("Process {0:d} using GPU {1:d}".format(config_num, device))
            environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(device)
        param_scores = {}
        train_data = np.load(join(out_path, "param_train_data.npy"))
        train_labels = np.load(join(out_path, "param_train_labels.npy"))
        val_data = np.load(join(out_path, "param_val_data.npy"))
        val_labels = np.load(join(out_path, "param_val_labels.npy"))
        session = None
        if device_queue is not None:
            session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=False,
                                                           gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                           log_device_placement=False))
            K.set_session(session)
        print("Training ", model_name, config_num, device)
        model_args = inspect.getargspec(model_obj.__init__).args
        if "index" in model_args:
            hail_model = model_obj(index=config_num, **params)
        else:
            hail_model = model_obj(**params)
        hail_model.fit(train_data,
                       train_labels)
        val_preds = hail_model.predict_proba(val_data)[:, 1]
        param_scores["Brier Skill Score"] = brier_skill_score(val_labels,
                                                              val_preds)
        param_scores["AUC"] = roc_auc_score(val_labels,
                                            val_preds)

        print("Scores ", config_num, device, param_scores["Brier Skill Score"], param_scores["AUC"])
        if device_queue is not None and device >= 0:
            session.close()
            del session
            device_queue.put(device)
        del hail_model
        return param_scores, config_num
    except Exception as e:
        if device_queue is not None and device >= 0:
            device_queue.put(device)
        print(traceback.format_exc())
        raise e


def train_best_sklearn_model(model_name, model_obj, best_combo, n, train_labels,
                             test_meta, test_labels, sample_scores, out_path):
    try:
        print("Train " + model_name)
        train_data = np.load(join(out_path, "best_train_data.npy"))
        test_data = np.load(join(out_path, "best_test_data.npy"))
        environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(0)
        session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=False,
                                                       gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                       log_device_placement=False))
        K.set_session(session)
        if model_name == "logistic_gan":
            hail_model = model_obj(index=n, **best_combo)
        else:
            hail_model = model_obj(**best_combo)
        hail_model.fit(train_data,
                       train_labels)
        print("Scoring " + model_name)
        test_preds = hail_model.predict_proba(test_data)[:, 1]
        test_pred_frame = test_meta.copy(deep=True)
        test_pred_frame["conv_net"] = test_preds
        test_pred_frame["label"] = test_labels
        test_pred_frame.to_csv(join(out_path, "predictions_{0}_sample_{1:03d}.csv".format(model_name, n)),
                               index_label="Index")
        sample_scores.loc[n, "Brier Score"] = brier_score(test_labels, test_preds)
        sample_scores.loc[n, "Brier Score Climo"] = brier_score(test_labels,
                                                                test_labels.mean())
        sample_scores.loc[n, "Brier Skill Score"] = brier_skill_score(test_labels, test_preds)
        sample_scores.loc[n, "AUC"] = roc_auc_score(test_labels, test_preds)
        if model_name == "logistic_gan":
            save_logistic_gan(hail_model, out_path)
        else:
            with open(join(out_path, "hail_{0}_sample_{1:03d}.pkl".format(model_name, n)), "wb") as model_file:
                pickle.dump(hail_model, model_file, pickle.HIGHEST_PROTOCOL)
        session.close()
        del session
        del hail_model
        return sample_scores
    except Exception as e:
        print(traceback.format_exc())
        raise e


def train_best_conv_net(best_combo, n, train_labels, test_meta, test_labels, sample_scores, out_path):
    try:
        print("Train Conv Net")
        train_data = np.load(join(out_path, "best_train_data.npy"))
        test_data = np.load(join(out_path, "best_test_data.npy"))
        environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(0)
        session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=False,
                                                        gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                        log_device_placement=False))
        K.set_session(session)
        hail_conv_net_model = hail_conv_net(**best_combo)
        hail_conv_net_model.fit(train_data,
                                train_labels,
                                batch_size=best_combo["batch_size"],
                                epochs=best_combo["num_epochs"], verbose=2)
        print("Scoring Conv Net")
        test_preds = hail_conv_net_model.predict(test_data).ravel()
        test_pred_frame = test_meta.copy(deep=True)
        test_pred_frame["conv_net"] = test_preds
        test_pred_frame["label"] = test_labels
        test_pred_frame.to_csv(join(out_path, "predictions_conv_net_sample_{0:03d}.csv".format(n)), index_label="Index")
        sample_scores.loc[n, "Brier Score"] = brier_score(test_labels, test_preds)
        sample_scores.loc[n, "Brier Score Climo"] = brier_score(test_labels,
                                                                test_labels.mean())
        sample_scores.loc[n, "Brier Skill Score"] = brier_skill_score(test_labels, test_preds)
        sample_scores.loc[n, "AUC"] = roc_auc_score(test_labels, test_preds)
        save_model(hail_conv_net_model, join(out_path, "hail_conv_net_sample_{0:03d}.h5".format(n)))
        session.close()
        del session
        del hail_conv_net_model
        return sample_scores
    except Exception as e:
        print(traceback.format_exc())
        raise e


def evaluate_conv_net(storm_norm_data, storm_meta,  hail_labels,
                      sampling_config, param_combos, out_path, num_gpus=8):
    """

    Args:
        storm_norm_data:
        storm_meta:
        hail_labels:
        sampling_config:
        param_combos:
        out_path:
        num_gpus:
    Returns:

    """
    unique_dates = np.unique(storm_meta["run_dates"])
    np.random.seed(sampling_config["random_seed"])
    storm_sampler = train_split_generator(unique_dates, sampling_config["train_split"],
                                          sampling_config["num_samples"])
    best_param_combos = []
    sample_scores = pd.DataFrame(index=np.arange(sampling_config["num_samples"]),
                                 columns=["Brier Score", "Brier Score Climo", "Brier Skill Score", "AUC"],
                                 dtype=float)
    for n in range(sampling_config["num_samples"]):
        environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        train_dates, test_dates = next(storm_sampler)
        print(train_dates, test_dates)
        train_indices = np.where(np.in1d(storm_meta["run_dates"], train_dates))[0]
        test_indices = np.where(np.in1d(storm_meta["run_dates"], test_dates))[0]
        all_members = np.unique(storm_meta.loc[train_indices, "members"])
        np.random.shuffle(all_members)
        member_split = int(np.round(all_members.size * sampling_config["member_split"]))
        train_members = all_members[:member_split]
        val_members = all_members[member_split:]
        print(train_members, val_members)
        train_member_indices = np.where(np.in1d(storm_meta.loc[train_indices, "members"], train_members))[0]
        val_member_indices = np.where(np.in1d(storm_meta.loc[train_indices, "members"], val_members))[0]
        param_scores = pd.DataFrame(index=np.arange(param_combos.shape[0]),
                                    columns=["Brier Skill Score", "AUC"], dtype=float)
        score_outputs = []

        param_train_data = storm_norm_data[train_indices][train_member_indices]
        param_train_labels = hail_labels[train_indices][train_member_indices]
        param_val_data = storm_norm_data[train_indices][val_member_indices]
        param_val_labels = hail_labels[train_indices][val_member_indices]
        print("Saving training data") 
        np.save(join(out_path, "param_train_data.npy"), param_train_data)
        np.save(join(out_path, "param_train_labels.npy"), param_train_labels)
        np.save(join(out_path, "param_val_data.npy"), param_val_data)
        np.save(join(out_path, "param_val_labels.npy"), param_val_labels)
        gpu_manager = Manager()
        gpu_queue = gpu_manager.Queue()
        n_pool = Pool(num_gpus, maxtasksperchild=1)
        for g in range(num_gpus):
            gpu_queue.put(g)
        
        for c in param_combos.index.values:
            print(c)
            score_outputs.append(n_pool.apply_async(train_single_conv_net,
                                                    (c, gpu_queue, param_combos.loc[c].to_dict(), out_path)))
        n_pool.close()
        n_pool.join()
        #for c in param_combos.index.values:
        #    score_outputs.append(train_single_conv_net(c, gpu_queue, param_combos.loc[c].to_dict(), out_path))
        for async_out in score_outputs:
            out = async_out.get()
            param_scores.loc[out[1]] = out[0]
        del n_pool
        del gpu_queue
        del gpu_manager
        best_config = param_scores["Brier Skill Score"].idxmax()
        best_combo = param_combos.loc[best_config].to_dict()
        param_scores.to_csv(join(out_path, "conv_net_param_scores_sample_{0:03d}.csv".format(n)),
                            index_label="Param Combo")
        best_param_combos.append(best_config)
        print("Best Config")
        print(param_combos.loc[best_config])
        pool = Pool(1)
        np.save(join(out_path, "best_train_data.npy"), storm_norm_data[train_indices])
        np.save(join(out_path, "best_test_data.npy"), storm_norm_data[test_indices])
        sample_scores = pool.apply(train_best_conv_net, (best_combo, n, 
                                                         hail_labels[train_indices],
                                                         storm_meta.loc[test_indices],
                                                         hail_labels[test_indices],
                                                         sample_scores, out_path))
        pool.close()
        pool.join()
        del pool
        sample_scores.to_csv(join(out_path, "conv_net_sample_scores.csv"), index_label="Sample")
    best_config_frame = param_combos.loc[best_param_combos]
    best_config_frame = best_config_frame.reset_index()
    best_config_frame.to_csv(join(out_path, "conv_net_best_params.csv"), index_label="Sample")
    return


def evaluate_sklearn_model(model_name, model_obj, storm_data, storm_meta, hail_labels,
                           sampling_config, param_combos, out_path, num_gpus=8):
    unique_dates = np.unique(storm_meta["run_dates"])
    np.random.seed(sampling_config["random_seed"])
    storm_sampler = train_split_generator(unique_dates, sampling_config["train_split"],
                                          sampling_config["num_samples"])
    best_param_combos = []
    sample_scores = pd.DataFrame(index=np.arange(sampling_config["num_samples"]),
                                 columns=["Brier Score", "Brier Score Climo", "Brier Skill Score", "AUC"],
                                 dtype=float)
    for n in range(sampling_config["num_samples"]):
        train_dates, test_dates = next(storm_sampler)
        train_indices = np.where(np.in1d(storm_meta["run_dates"], train_dates))[0]
        test_indices = np.where(np.in1d(storm_meta["run_dates"], test_dates))[0]
        all_members = np.unique(storm_meta.loc[train_indices, "members"])
        np.random.shuffle(all_members)
        member_split = int(np.round(all_members.size * sampling_config["member_split"]))
        train_members = all_members[:member_split]
        val_members = all_members[member_split:]
        train_member_indices = np.where(np.in1d(storm_meta.loc[train_indices, "members"], train_members))[0]
        val_member_indices = np.where(np.in1d(storm_meta.loc[train_indices, "members"], val_members))[0]
        param_scores = pd.DataFrame(index=np.arange(param_combos.shape[0]), 
                                    columns=["Brier Skill Score", "AUC"], dtype=float)

        score_outputs = []
        param_train_data = storm_data[train_indices][train_member_indices]
        param_train_labels = hail_labels[train_indices][train_member_indices]
        param_val_data = storm_data[train_indices][val_member_indices]
        param_val_labels = hail_labels[train_indices][val_member_indices]
        print("Saving training data")
        np.save(join(out_path, "param_train_data.npy"), param_train_data)
        np.save(join(out_path, "param_train_labels.npy"), param_train_labels)
        np.save(join(out_path, "param_val_data.npy"), param_val_data)
        np.save(join(out_path, "param_val_labels.npy"), param_val_labels)
        gpu_manager = Manager()
        gpu_queue = gpu_manager.Queue()
        n_pool = Pool(num_gpus, maxtasksperchild=1)
        for g in range(num_gpus):
            gpu_queue.put(g)

        for c in param_combos.index.values:
            print(c)
            score_outputs.append(n_pool.apply_async(train_single_sklearn_model,
                                                    (model_name, model_obj, c, param_combos.loc[c].to_dict(),
                                                     out_path),
                                                    dict(device_queue=gpu_queue)))
        n_pool.close()
        n_pool.join()
        for async_out in score_outputs:
            out = async_out.get()
            param_scores.loc[out[1]] = out[0]
        del n_pool
        del gpu_queue
        del gpu_manager
        #for c in param_combos.index:
        #    print(param_combos.loc[c])
        #    model_inst = model_obj(**param_combos.loc[c].to_dict())
        #    model_inst.fit(storm_data[train_indices][train_member_indices],
        #                   hail_labels[train_indices][train_member_indices])
        #    val_preds = model_inst.predict_proba(storm_data[train_indices][val_member_indices])[:, 1]
        #    param_scores.loc[c, "Brier Skill Score"] = brier_skill_score(hail_labels[train_indices][val_member_indices],
        #                                                                 val_preds)
        #    param_scores.loc[c, "AUC"] = roc_auc_score(hail_labels[train_indices][val_member_indices],
        #                                               val_preds)
        #    if param_scores.loc[c, "Brier Skill Score"] > best_score:
        #        best_config = c
        #        best_score = param_scores.loc[c, "Brier Skill Score"]
        #    del model_inst

        param_scores.to_csv(join(out_path, "{0}_param_scores_sample_{1:03d}.csv".format(model_name, n)),
                            index_label="Param Combo")
        best_config = param_scores["Brier Skill Score"].idxmax()
        best_combo = param_combos.loc[best_config].to_dict()
        best_param_combos.append(best_config)
        print("Best Config")
        print(param_combos.loc[best_config])
        pool = Pool(1)
        np.save(join(out_path, "best_train_data.npy"), storm_data[train_indices])
        np.save(join(out_path, "best_test_data.npy"), storm_data[test_indices])
        sample_scores = pool.apply(train_best_sklearn_model, (model_name, model_obj, best_combo, n,
                                                              hail_labels[train_indices],
                                                              storm_meta.loc[test_indices],
                                                              hail_labels[test_indices],
                                                              sample_scores, out_path))
        pool.close()
        pool.join()
        del pool
        sample_scores.to_csv(join(out_path, "{0}_sample_scores.csv".format(model_name)), index_label="Sample")
        #print("Train Best " + model_name)
        #model_inst = model_obj(**param_combos.loc[best_config].to_dict())
        #model_inst.fit(storm_data[train_indices],
        #               hail_labels[train_indices])
        #print("Scoring " + model_name)
        #test_pred_frame = storm_meta.loc[test_indices]
        #test_pred_frame[model_name] = model_inst.predict_proba(storm_data[test_indices])[:, 1]
        #test_pred_frame["label"] = hail_labels[test_indices]
        #test_preds = test_pred_frame[model_name].values
        #test_pred_frame = pd.DataFrame({"indices": test_indices,
                                       # "lon": storm_centers[test_indices, 0],
                                       # "lat": storm_centers[test_indices, 1],
                                       # "run_dates": storm_run_dates[test_indices],
                                       # "valid_dates": storm_valid_dates[test_indices],
                                       # "members": storm_members[test_indices],
                                       # model_name: test_preds,
                                       # "label": hail_labels[test_indices]},
                                       #columns=["indices", "lon", "lat", "dates", "members", "conv_net", "label"])
        #test_pred_frame.to_csv(join(out_path, "predictions_{0}_sample_{1:03d}.csv".format(model_name, n)), index_label="Index")
        #sample_scores.loc[n, "Brier Score"] = brier_score(hail_labels[test_indices], test_preds)
        #sample_scores.loc[n, "Brier Score Climo"] = brier_score(hail_labels[test_indices],
        #                                                        hail_labels[test_indices].mean())
        #sample_scores.loc[n, "Brier Skill Score"] = brier_skill_score(hail_labels[test_indices], test_preds)
        #sample_scores.loc[n, "AUC"] = roc_auc_score(hail_labels[test_indices], test_preds)
        #
        #del model_inst
    #sample_scores.to_csv(join(out_path, "{0}_sample_scores.csv".format(model_name)), index_label="Sample")
    best_config_frame = param_combos.loc[best_param_combos]
    best_config_frame = best_config_frame.reset_index()
    best_config_frame.to_csv(join(out_path, "{0}_best_params.csv".format(model_name)), index_label="Sample")
    return





if __name__ == "__main__":
    main()
