import os
from datasets import DownloadConfig, load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

download_config = DownloadConfig(
    force_download=False,
    resume_download=True,
    extract_compressed_file=True,
)

def transform_fake_ds(ds):
    ds = ds.remove_columns([col for col in ds.column_names if col not in ["png", "model.txt"]])
    ds = ds.rename_columns({"png": "image", "model.txt": "generator"})
    ds = ds.add_column("label", [1] * len(ds))
    return ds

def transform_real_ds(ds):
    ds = ds.remove_columns([col for col in ds.column_names if col != "png"])
    ds = ds.add_column("generator", ["Real"] * len(ds))
    ds = ds.rename_columns({"png": "image"})
    ds = ds.add_column("label", [0] * len(ds))
    return ds

def build_dataset():
    fake_ds = load_dataset(
        "lesc-unifi/dragon",
        "Large",
        download_config=download_config
    )

    fake_ds = fake_ds["train"]
    train_fake_ds, val_fake_ds = train_test_split(
        train_fake_ds, test_size=0.15, random_state=42, shuffle=True
    )
    test_fake_ds = fake_ds["test"]

    real_ds = load_dataset(
        "timm/imagenet-1k-wds",
        split=["train[:50%]","validation","test"],
        download_config=download_config)

    train_real_ds = real_ds[0]
    val_real_ds = real_ds[1]
    test_real_ds = real_ds[2]

    fake = [train_fake_ds, val_fake_ds, test_fake_ds]
    real = [train_real_ds, val_real_ds, test_real_ds]

    for i in range(3):
        fake[i] = transform_fake_ds(fake[i])
        real[i] = transform_real_ds(real[i])

    complete_datasets = {
        "train": concatenate_datasets([fake[0], real[0]]).shuffle(seed=42),
        "val": concatenate_datasets([fake[1], real[1]]).shuffle(seed=42),
        "test": concatenate_datasets([fake[2], real[2]]).shuffle(seed=42)
    }
    return complete_datasets
    