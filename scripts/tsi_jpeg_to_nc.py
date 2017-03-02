from deepsky.image import load_raw_images_date, clear_image_files
import pandas as pd
import argparse
from multiprocessing import Pool
from os.path import join
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", help="Start Date")
    parser.add_argument("-e", "--end", help="Start Date")
    parser.add_argument("-i", "--input", help="Input path")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-p", "--proc", default=1, type=int, help="Number of processors")
    args = parser.parse_args()
    pool = Pool(args.proc)
    dates = pd.DatetimeIndex(start=args.start, end=args.end, freq="1D")
    for date in dates:
        pool.apply_async(convert_tsi_jpeg_to_nc, (date, args.input, args.output))
    pool.close()
    pool.join()
    return


def convert_tsi_jpeg_to_nc(date, in_path, out_path):
    try:
        image_data = load_raw_images_date(date, in_path)
        print("Saving to netCDF {0}".format(date.strftime("%Y-%m-%d")))
        image_data.to_dataset(name="tsi_image").to_netcdf(join(out_path,
                                                               "tsi_sgp_{0}.nc".format(date.strftime("%Y%m%d"))),
                                                          encoding={"tsi_image": {"zlib":True, "complevel": 1,
                                                                                  "chunksizes":(120, 32, 32)}})
        clear_image_files(date, in_path)
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return

if __name__ == "__main__":
    main()
