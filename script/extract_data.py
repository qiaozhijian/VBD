import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# set tf to cpu only
tf.config.set_visible_devices([], "GPU")
import jax

jax.config.update("jax_platform_name", "cpu")

import glob
import argparse
import pickle
from vbd.data.data_utils import *
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

from waymax import dataloader
from waymax.config import DataFormat
import functools

MAX_NUM_OBJECTS = 64
MAX_POLYLINES = 256
MAX_TRAFFIC_LIGHTS = 16
CURRENT_INDEX = 10
NUM_POINTS_POLYLINE = 30


def data_process(
    data_dir: str,
    save_dir: str,
    save_raw: bool = False,
    only_raw: bool = False,
):
    """
    Process the Waymax dataset and save the processed data.

    Args:
        data_dir (str): Directory path of the Waymax dataset.
        save_dir (str): Directory path to save the processed data.
        save_raw (bool, optional): Whether to save the raw scenario data. Defaults to False.
    """
    # Waymax Dataset
    tf_dataset = dataloader.tf_examples_dataset(
        path=data_dir,
        data_format=DataFormat.TFRECORD,
        preprocess_fn=tf_preprocess,
        repeat=1,
        # num_shards=16,
        deterministic=True,
    )

    tf_dataset_iter = tf_dataset.as_numpy_iterator()

    os.makedirs(save_dir, exist_ok=True)

    for example in tf_dataset_iter:

        scenario_id_binary, scenario = tf_postprocess(example)
        scenario_id = scenario_id_binary.tobytes().decode("utf-8")

        scenario_filename = os.path.join(save_dir, "scenario_" + scenario_id + ".pkl")

        # check if file exists
        if os.path.exists(scenario_filename):
            continue

        if only_raw:
            data_dict = {"scenario_raw": scenario}
        else:
            data_dict = data_process_scenario(
                scenario,
                max_num_objects=MAX_NUM_OBJECTS,
                max_polylines=MAX_POLYLINES,
                current_index=CURRENT_INDEX,
                num_points_polyline=NUM_POINTS_POLYLINE,
            )
            if save_raw:
                data_dict["scenario_raw"] = scenario

        data_dict["scenario_id"] = scenario_id

        with open(scenario_filename, "wb") as f:
            pickle.dump(data_dict, f)


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/Dataset/Waymo/V1_2_tf")
    parser.add_argument("--save_dir", type=str, default="/data/Dataset/Waymo/VBD")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--only_raw", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Processing data from {args.data_dir} and Saving to {args.save_dir}")

    def process(save_raw=False, only_raw=False):
        """
        Process a specific dataset and save the processed data.

        Args:
            dataset (str): Name of the dataset to process.
            save_raw (bool, optional): Whether to save the raw scenario data. Defaults to False.
        """
        data_files = glob.glob(args.data_dir + "/*")
        if args.only_raw:
            save_dir = os.path.join(args.save_dir, "extracted")
        else:
            save_dir = os.path.join(args.save_dir, "processed")

        n_files = len(data_files)
        print(f"Processing {n_files} files in {args.data_dir}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving to {save_dir}")

        data_process_partial = functools.partial(
            data_process,
            save_dir=save_dir,
            save_raw=save_raw,
            only_raw=only_raw,
        )
        process_map(data_process_partial, data_files, max_workers=args.num_workers)

    process(save_raw=args.save_raw, only_raw=args.only_raw)
