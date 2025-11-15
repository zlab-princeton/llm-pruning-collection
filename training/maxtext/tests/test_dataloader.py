import jax
import glob
from pathlib import Path
import grain.python as grain

from jax.sharding import Mesh
from jax.experimental import mesh_utils
from MaxText.input_pipeline import input_pipeline_interface

from MaxText import max_logging

import tensorflow as tf

def find_data_files(data_file_pattern):
    data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
    assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
    max_logging.log(f"Found {len(data_files)} files for train/eval with grain")
    return data_files

def decode_arrayrecord(raw_bytes):
    example = tf.train.Example()
    example.ParseFromString(raw_bytes)
    return {k: v.bytes_list.value if v.bytes_list.value else v.int64_list.value
            for k, v in example.features.feature.items()}

def get_datasets(
    data_file_pattern,
    data_file_type,
    shuffle,
    shuffle_seed,
    num_epoch,
    dataloading_host_index,
    dataloading_host_count,
    grain_worker_count,
):
    """Load dataset from array_record files for using with grain"""
    if data_file_type == "arrayrecord":
        if ";" in data_file_pattern:
            data_file_patterns, weights = zip(*[pattern.split(":") for pattern in data_file_pattern.split(";")])
            assert len(data_file_patterns) == len(weights), "Number of data file patterns and weights must match"
            weights = [float(weight) for weight in weights]
            weights = [round(weight / sum(weights), 4) for weight in weights]
            dataset_list = [
                grain.MapDataset.source(grain.ArrayRecordDataSource(find_data_files(pattern))) for pattern in data_file_patterns
            ]
            dataset = grain.MapDataset.mix(dataset_list, weights)
        else:
            data_files = find_data_files(data_file_pattern)
            dataset = grain.MapDataset.source(grain.ArrayRecordDataSource(data_files))
        if shuffle:
            dataset = dataset.shuffle(seed=shuffle_seed)
        dataset = dataset.repeat(num_epoch)
        dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
        dataset = dataset.to_iter_dataset()
        # import pdb; pdb.set_trace()
    elif data_file_type == "parquet":
        data_files = find_data_files(data_file_pattern)
        dataset = grain.MapDataset.source(data_files)
        if shuffle:
            dataset = dataset.shuffle(seed=shuffle_seed)
        dataset = dataset.repeat(num_epoch)
        dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
        assert grain_worker_count <= len(dataset), (
            f"grain worker count is currently {grain_worker_count}, exceeding the max allowable value {len(dataset)} "
            f"(file shard count of a data loading host) for your dataset. "
            f"Please lower grain_worker_count or increase file shard count."
        )
        dataset = dataset.map(grain.experimental.ParquetIterDataset)
        dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=len(dataset))
        dataset = grain.experimental.WindowShuffleIterDataset(dataset, window_size=100, seed=shuffle_seed)
    else:
        raise ValueError(f"grain pipeline supports (arrayrecord, parquet) as grain_file_type, but got {data_file_type}")

    return dataset

if __name__ == '__main__':
    mesh_shape_1d = (len(jax.devices()),)
    mesh_axes = ['data']
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), mesh_axes)
    #   global_batch_size_to_load = int(micro_batch_size_to_load * gradient_accumulation_steps)
    #   global_batch_size_to_train_on = int(micro_batch_size_to_train_on * gradient_accumulation_steps)
    process_indices = input_pipeline_interface.get_process_loading_real_data(
        ['data'],
        512,
        512,
        8192,
        mesh
    )
        
    dataset = get_datasets(
        # '/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record',
        '/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/dclm_baseline_1.0.chunk.00000.array_record',
        'arrayrecord',
        shuffle=False,
        shuffle_seed=0,
        num_epoch=1,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        grain_worker_count=1,
    )
    
    raw = dataset._parent[0]   # the bytes you printed
    decoded = decode_arrayrecord(raw)
    print(decoded.keys())
    print(decoded["text"][:200])