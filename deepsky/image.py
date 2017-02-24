import numpy as np
from skimage.io import imread
import xarray as xr
from glob import glob
import os
from os.path import join
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
        image_tar_file = image_tar_files[0]
        tf = tarfile.open(image_tar_file)
        out_path = join(path,date.strftime("%Y%m%d")) 
        os.mkdir(out_path)
        tf.extractall(path=out_path)
        tf.close()
        image_files = pd.Series(sorted(glob(out_path + "*.jpg")))
        image_dates = pd.DatetimeIndex(image_files.str.split(".").str[-2])
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
        data = xr.DataArray(image_data, coords={"time": image_dates, "y": rows, "x": cols, "color":np.arange(3)}, dims=("time", "y", "x", "color"), attrs={"long_name":"TSI image", "units":""})
    else:
        raise IOError(date + " not found")
    return data


def clear_image_files(date, path):
    full_path = join(path, date.strftime("%Y%m%d"))
    image_files = sorted(glob(full_path + "*.jpg"))
    for image_file in image_files:
        os.remove(image_file)
    os.rmdir(full_path)
