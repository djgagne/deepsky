import numpy as np
import pandas as pd
import yaml
import argparse
from os import environ
from os.path import join
from deepsky.data import load_storm_patch_data
from deepsky.gan import normalize_multivariate_data
from deepsky.importance import variable_importance
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
    parser.add_argument("-e", "--emb", action="store_true", help="Calculate and visualize model embedding")
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
    print("Loading data")
    global storm_norm_data, storm_meta, storm_flat_data, storm_mean_data, hail_labels
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
    if args.imp:
        importance_manager(out_path, model_names, input_variables, sampling_config["num_samples"],
                           num_permutations, score_funcs, args.proc)
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
        session.close()
        device_queue.put(device)
    except Exception as e:
        print(traceback.format_exc())
        if device is not None:
            device_queue.put(device)
        raise e
    return



if __name__ == "__main__":
    main()
