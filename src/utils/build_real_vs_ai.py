from datasets import load_dataset, concatenate_datasets, DatasetDict, DownloadConfig

HF_CACHE = "/home/jupyter/Data/dragon_cache/huggingface"
DS_CACHE = "/home/jupyter/Data/dragon_cache/datasets"

cfg = DownloadConfig(cache_dir=HF_CACHE, resume_download=True, extract_compressed_file=True)

def build(split_frac="train[:50%]"):
    ai   = load_dataset("lesc-unifi/dragon",        "Large",
                        split=["train","validation","test"],
                        download_config=cfg)
    real = load_dataset("timm/imagenet-1k-wds", split=[split_frac,"validation","test"],
                        download_config=cfg)

    # tag with labels  ── 0 = real, 1 = ai
    ai   = [s.map(lambda _: {"label": 1}) for s in ai]
    real = [s.map(lambda _: {"label": 0}) for s in real]

    mixed = DatasetDict({
        "train":      concatenate_datasets([ai[0],   real[0]]).shuffle(seed=42),
        "validation": concatenate_datasets([ai[1],   real[1]]).shuffle(seed=42),
        "test":       concatenate_datasets([ai[2],   real[2]]),
    })
    return mixed