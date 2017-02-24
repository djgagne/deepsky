from ..deepsky.image import *
import numpy as np
import pandas as pd
import xarray as xr
import argparse
from multiprocessing import Pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", help="Start Date")
    parser.add_argument("-e", "--end", help="Start Date")
    parser.add_argument("-i", "--in", help="Input path")
    parser.add_argument("-o", "--out", help="Output path")
    parser.add_argument("-p", "--proc", default=1, type=int, help="Number of processors")
    args = parser.parse_args()
    pool = Pool(args.proc)
    dates = pd.DatetimeIndex()
    return


if __name__ == "__main__":
    main()
