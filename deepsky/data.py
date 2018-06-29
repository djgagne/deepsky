import numpy as np
import pandas as pd
import xarray as xr
import traceback
from os.path import join
from glob import glob
from multiprocessing import Pool


def load_storm_data_file(data_file, variable_names):
    try:
        run_filename = data_file.split("/")[-1][:-3].split("_")
        member = int(run_filename[6])
        run_date = run_filename[4]
        ds = xr.open_dataset(data_file)
        patch_arr = []
        all_vars = list(ds.variables.keys())
        meta_cols = ["center_lon", "center_lat", "valid_dates", "run_dates", "members"]
        return_dict = {"data_file": data_file, "meta": None, "data_patches": None}
        if np.all(np.in1d(variable_names, all_vars)):
            meta_dict = {}
            meta_dict["center_lon"] = ds["longitude"][:, 32, 32].values
            meta_dict["center_lat"] = ds["latitude"][:, 32, 32].values
            meta_dict["valid_dates"] = pd.DatetimeIndex(ds["valid_date"].values)
            meta_dict["run_dates"] = np.tile(run_date, meta_dict["valid_dates"].size)
            meta_dict["members"] = np.tile(member, meta_dict["valid_dates"].size)
            return_dict["meta"] = pd.DataFrame(meta_dict, columns=meta_cols)
            for variable in variable_names:
                patch_arr.append(ds[variable][:, 16:-16, 16:-16].values.astype(np.float32))
            return_dict["data_patches"] = np.stack(patch_arr, axis=-1)
            print(data_file, return_dict["meta"].size)
        ds.close()
        del patch_arr[:]
        del patch_arr
        del ds
        return return_dict
    except Exception as e:
        print(traceback.format_exc())
        raise e


def load_storm_patch_data(data_path, variable_names, n_procs):
    """


    Args:
        data_path:
        variable_names:
        n_procs:

    Returns:

    """
    data_patches = []
    data_meta = []

    data_files = sorted(glob(join(data_path, "*.nc")))
    pool = Pool(n_procs, maxtasksperchild=1)
    file_check = data_files[:]

    def combine_storm_data_files(return_obj):
        f_index = file_check.index(return_obj["data_file"])
        if return_obj["meta"] is not None:
            data_patches[f_index] = return_obj["data_patches"]
            data_meta[f_index] = return_obj["meta"]
        else:
            file_check.pop(f_index)
            data_patches.pop(f_index)
            data_meta.pop(f_index)

    for data_file in data_files:
        data_patches.append(None)
        data_meta.append(None)
        pool.apply_async(load_storm_data_file, (data_file, variable_names), callback=combine_storm_data_files)
    pool.close()
    pool.join()
    del pool
    all_data = np.vstack(data_patches)
    all_meta = pd.concat(data_meta, ignore_index=True)
    return all_data, all_meta