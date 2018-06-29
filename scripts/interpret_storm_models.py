import numpy as np
import pandas as pd
import yaml
import argparse
from os import environ
from os.path import join
from deepsky.data import load_storm_patch_data
from deepsky.gan import normalize_multivariate_data
from deepsky.importance import variable_importance, activated_analogs
from deepsky.metrics import brier_skill_score, roc_auc
from deepsky.models import load_logistic_gan
from multiprocessing import Pool, Manager
import traceback
import keras.backend as K
import pickle
from keras.models import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("-i", "--imp", action="store_true", help="Calculate Variable Importance")
    parser.add_argument("-e", "--enc", action="store_true", help="Calculate and visualize model encoding")
    parser.add_argument("-m", "--max", action="store_true", help="Find examples that maximize activations")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    output_config = config["output"]
    sampling_config = config["sampling"]
    data_path = config["data_path"]
    input_variables = config["input_variables"]
    out_path = config["out_path"]
    num_permutations = config["num_permutations"]
    model_names = config["model_names"]
    encoding = config["encoding"]
    print("Loading data")
    global storm_norm_data, storm_meta, storm_flat_data, storm_mean_data, hail_labels, max_hail
    storm_scaling_values = pd.read_csv(join(out_path, "scaling_values.csv"), index_col="Index")
    storm_data, storm_meta = load_storm_patch_data(data_path, input_variables, args.proc)
    storm_norm_data, storm_scaling_values = normalize_multivariate_data(storm_data,
                                                                        scaling_values=storm_scaling_values)
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
    score_funcs = {"auc": roc_auc, "bss": brier_skill_score}
    batch_size = 128
    if args.imp:
        importance_manager(out_path, model_names, input_variables, sampling_config["num_samples"],
                           num_permutations, score_funcs, args.proc)
    if args.enc:
        encoding_manager(out_path, model_names, num_permutations, encoding, args.proc, batch_size)
    return


def importance_manager(output_dir, model_names, input_variables, num_models, num_permutations, score_funcs, num_procs):
    """
    Manages the parallel calculation of variable importance scores for all storm models.

    Args:
        output_dir: Path to location of model files
        model_names: List of model names
        input_variables: List of input variables
        num_models: Number of model samples trained
        num_permutations: Number of variable importance permutations calculated
        score_funcs: Dictionary of scoring functions with name->function pairs. Score function should accept (obs, fore)
            arguments in that order.
        num_procs: Number of processors used for scoring

    Returns:

    """
    gpu_manager = Manager()
    gpu_queue = gpu_manager.Queue()
    for g in range(num_procs):
        gpu_queue.put(g)
    pool = Pool(num_procs, maxtasksperchild=1)
    for model_name in model_names:
        for model_number in range(num_models):
            pool.apply_async(importance_model, (model_name, model_number,
                                                gpu_queue, input_variables, num_permutations,
                                                score_funcs, output_dir))
    pool.close()
    pool.join()
    return


def importance_model(model_name, model_number, device_queue, input_variables,
                     num_permutations, score_funcs, output_dir):
    device = None
    try:
        device = int(device_queue.get())
        print("Process {0} {1:d} using GPU {2:d}".format(model_name, model_number, device))
        if model_name in ["conv_net", "logistic_gan"]:
            environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(device)
            session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=False,
                                                       gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                       log_device_placement=False))
            K.set_session(session)
        if model_name == "conv_net":
            model = load_model(join(output_dir, "hail_conv_net_sample_{0:03d}.h5".format(model_number)))
            sklearn_model = False
        elif model_name == "logistic_gan":
            model = load_logistic_gan(output_dir, model_number)
            sklearn_model = True
        else:
            with open(join(output_dir, "hail_{0}_sample_{1:03d}.pkl".format(model_name,
                                                                            model_number)), "rb") as model_pickle:
                model = pickle.load(model_pickle)
            sklearn_model = True
        sample_preds = pd.read_csv(join(output_dir, "predictions_conv_net_sample_{0:03d}.csv".format(model_number)),
                                   index_col="Index")
        test_dates = sample_preds["run_dates"].unique().astype("U10")
        all_dates = storm_meta["run_dates"].unique().astype("U10")
        train_dates = all_dates[~np.isin(all_dates, test_dates)]
        train_indices = np.where(np.in1d(storm_meta["run_dates"].values, train_dates))[0]
        var_scores = dict()
        if "mean" in model_name:
            mean_model = True
            imp_data = storm_mean_data[train_indices]
        elif "pca" in model_name:
            mean_model = True
            imp_data = storm_flat_data[train_indices]
            print("Using flat data")
        else:
            mean_model = False
            imp_data = storm_norm_data[train_indices]
        for score_name, score_func in score_funcs.items():
            var_scores[score_name] = variable_importance(imp_data,
                                                         hail_labels[train_indices],
                                                         input_variables,
                                                         model_name,
                                                         model,
                                                         score_func,
                                                         permutations=num_permutations,
                                                         sklearn_model=sklearn_model,
                                                         mean_model=mean_model)
            var_scores[score_name].to_csv(join(output_dir, "var_importance_{0}_{1}_{2:03d}.csv".format(model_name,
                                                                                                       score_name,
                                                                                                       model_number)),
                                          index_label="Index")
        if model_name in ["conv_net", "logistic_gan"]:
            session.close()
        device_queue.put(device)
    except Exception as e:
        print(traceback.format_exc())
        if device is not None:
            device_queue.put(device)
        raise e
    return


def encoding_manager(output_dir, model_names, num_models, enc_2d_methods, num_procs, batch_size):
    pool = Pool(num_procs, maxtasksperchild=1)
    gpu_manager = Manager()
    gpu_queue = gpu_manager.Queue()
    for g in range(num_procs):
        gpu_queue.put(g)
    for model_number in range(num_models):
        for model_name in model_names:
            pool.apply_async(model_encoder, (model_name, model_number, gpu_queue, enc_2d_methods, output_dir, batch_size))
    pool.close()
    pool.join()


def model_encoder(model_name, model_number, device_queue, enc_2d_methods, output_dir, batch_size):
    device = None
    try:
        device = -1 
        if model_name in ["conv_net", "logistic_gan"]:
            device = int(device_queue.get())
            environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(device)
            session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=False,
                                                           gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                           log_device_placement=False))
            K.set_session(session)
        print("Process {0} {1:d} using GPU {2:d}".format(model_name, model_number, device))
        if model_name == "conv_net":
            model = load_model(join(output_dir, "hail_conv_net_sample_{0:03d}.h5".format(model_number)))
            pred_func = K.function([model.input, K.learning_phase()], [model.layers[-3].output]) 
            print(model_name, model_number, "Generating encoding")
            batch_indices = list(range(0, storm_norm_data.shape[0], batch_size)) + [storm_norm_data.shape[0]]
            enc_batches = []
            for b, bi in enumerate(batch_indices[:-1]):
                enc_batches.append(pred_func([storm_norm_data[bi: batch_indices[b + 1]], 0])[0])
            encoding = np.vstack(enc_batches)
            print(model_name, model_number, encoding.shape)
            print(model_name, model_number, "Generating prediction")
            prediction = model.predict(storm_norm_data)[:, 0]
        elif model_name == "logistic_gan":
            model = load_logistic_gan(output_dir, model_number)
            print(model_name, model_number, "Generating encoding")
            encoding = model.encoder.predict(storm_norm_data)
            print(model_name, model_number, encoding.shape)
            print(model_name, model_number, "Generating prediction")
            prediction = model.predict_proba(storm_norm_data)[:, 1]
        else:
            with open(join(output_dir, "hail_{0}_sample_{1:03d}.pkl".format(model_name,
                                                                            model_number)), "rb") as model_pickle:
                model = pickle.load(model_pickle)
            if model_name == "logistic_pca":
                print(model_name, model_number, "Generating encoding")
                encoding = model.transform(storm_flat_data)
                print(model_name, model_number, encoding.shape)
                print(model_name, model_number, "Generating prediction")
                prediction = model.predict_proba(storm_flat_data)[:, 1]
            else:
                print(model_name, model_number, "Generating encoding")
                encoding = storm_mean_data
                print(model_name, model_number, encoding.shape)
                print(model_name, model_number, "Generating prediction")
                prediction = model.predict_proba(storm_mean_data)[:, 1]
        sample_preds = pd.read_csv(join(output_dir, "predictions_conv_net_sample_{0:03d}.csv".format(model_number)),
                                   index_col="Index")
        test_dates = sample_preds["run_dates"].unique().astype("U10")
        all_dates = storm_meta["run_dates"].unique().astype("U10")
        train_dates = all_dates[~np.isin(all_dates, test_dates)]
        train_indices = np.where(np.in1d(storm_meta["run_dates"].values, train_dates), 1, 0)
        encoding_cols = ["E{0:03d}".format(e) for e in range(encoding.shape[1])]
        en_frame = pd.DataFrame(encoding, columns=encoding_cols)
        #enc_2d_data = []
        #for enc_name, enc_params in enc_2d_methods.items():
        #    print(model_name, model_number, enc_name)
        #    if enc_name == "tsne":
        #        enc_2d_model = TSNE(**enc_params)
        #    elif enc_name == "isomap":
        #        enc_2d_model = Isomap(**enc_params)
        #    elif enc_name == "lle":
        #        enc_2d_model = LocallyLinearEmbedding(**enc_params)
        #    elif enc_name == "mds":
        #        enc_2d_model = MDS(**enc_params)
        #    elif enc_name == "pca":
        #        enc_2d_model = PCA(**enc_params)
        #    else:
        #        print(enc_name + " not supported.")
        #        enc_2d_model = None
        #    if enc_2d_model is not None:
        #        enc_2d_cols = [enc_name + "_{0:02d}".format(c) for c in range(enc_2d_model.n_components)]
        #        enc_2d_data.append(pd.DataFrame(enc_2d_model.fit_transform(encoding), columns=enc_2d_cols))
        #
        print(model_name, model_number, "Merging data")
        label_frame = pd.DataFrame(np.column_stack((train_indices, prediction, hail_labels, max_hail)),
                                   columns=["is_training",
                                            "{0}_{1:03d}_prob_severe_hail".format(model_name, model_number),
                                            "is_severe_hail", "max_hail_size"])
        out_frame = pd.concat([storm_meta, en_frame, label_frame], axis=1)
        print(model_name, model_number, "Saving data")
        out_frame.to_csv(join(output_dir, "encoding_{0}_{1:03d}.csv".format(model_name, model_number)),
                         index_label="Index")
    except Exception as e:
        print(traceback.format_exc())
        if device is not None:
            device_queue.put(device)
        raise e


def interpret_neuron_activations(model_name, model_number, device_queue, output_dir, num_analogs,
                                 dense_layer_index=-2, conv_layer_index=-6):
    device = int(device_queue.get())
    environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(device)
    session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=False,
                                                   gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                   log_device_placement=False))
    K.set_session(session)
    model = load_model(join(output_dir, "hail_conv_net_sample_{0:03d}.h5".format(model_number)))
    for i in range(4):
        for j in range(4):
            combined_info, top_gradients = activated_analogs(storm_norm_data, model, num_analogs,
                                                             filter_index=(i, j),
                                                             dense_layer_index=dense_layer_index,
                                                             conv_layer_index=conv_layer_index)
            combined_info.to_csv(join(output_dir, "activated_analogs_{0}_{1:03d}_{2:d}_{3:d}.csv".format(model_name,
                                                                                                         model_number,
                                                                                                         i,
                                                                                                         j)),
                                 index_col="Index")
            np.save(join(output_dir, "activated_gradients_{0}_{1:03d}_{2:d}_{3:d}.npy".format(model_name,
                                                                                              model_number,
                                                                                              i, j)),
                    top_gradients)
    return
if __name__ == "__main__":
    main()
