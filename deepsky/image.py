import numpy as np
from skimage.io import imread
import xarray as xr
from glob import glob
import os
from os.path import join, exists
import tarfile
import pandas as pd


def load_raw_images_date(date_in, path, image_diameter=480, row_start=75, max_radius=230):
    """
    Loads images for a given date from the tar file containing those images. The files are first untarred, and then they are loaded into
    an xarray DataArray. In the process, non-sky areas in the image are cropped and zeroed out.
    """
    date = pd.Timestamp(date_in)
    image_tar_files = glob(path + "*.{0}.*.jpg.tar".format(date.strftime("%Y%m%d")))
    data = None
    if len(image_tar_files) == 1:
        print("Loading data from {0}".format(date.strftime("%Y-%m-%d")))
        image_tar_file = image_tar_files[0]
        tf = tarfile.open(image_tar_file)
        tar_names = pd.Series(tf.getnames())
        tar_dates = pd.DatetimeIndex(tar_names.str.split(".").str[-2])
        valid_files = tar_names.values[np.where(((tar_dates.minute == 0) | (tar_dates.minute == 30)) & (tar_dates.second == 0))]
        out_path = join(path,date.strftime("%Y%m%d"))
        if not exists(out_path):
            os.mkdir(out_path)
        for valid_file in valid_files:
            tf.extract(valid_file, path=out_path)
        tf.close()
        image_files = pd.Series(sorted(glob(join(out_path, "*.jpg"))))
        image_dates = pd.DatetimeIndex(image_files.str.split(".").str.get(-2))
        rows = np.arange(image_diameter)
        cols = np.arange(image_diameter)
        row_grid, col_grid = np.meshgrid(rows, cols)
        distance_from_center = np.sqrt((row_grid - image_diameter/2) ** 2 + (col_grid - image_diameter / 2) ** 2)
        distance_filter = np.repeat(np.where(distance_from_center < max_radius, 1, 0).reshape(image_diameter, 
                                                                                              image_diameter, 
                                                                                              1), 3, axis=2).astype("uint8")
        image_data = np.zeros((len(image_files), image_diameter, image_diameter, 3), dtype="uint8")
        for i, image_file in enumerate(image_files):
            image_data[i] = imread(image_file)[row_start:row_start + image_diameter] * distance_filter
        data = xr.DataArray(image_data, coords={"time": image_dates, "y": rows, "x": cols, "color":np.arange(3)},
                            dims=("time", "y", "x", "color"), attrs={"long_name":"TSI image", "units":""})
    else:
        raise IOError(date.strftime("%Y%m%d") + " not found")
    return data


def clear_image_files(date, path):
    print("Removing jpeg files from {0}".format(date.strftime("%Y-%m-%d")))
    full_path = join(path, date.strftime("%Y%m%d"))
    image_files = sorted(glob(join(full_path,"*.jpg")))
    for image_file in image_files:
        os.remove(image_file)
    os.rmdir(full_path)
