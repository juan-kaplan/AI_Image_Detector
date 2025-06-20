import os
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_splits(
    fake_images_dir: str,
    real_images_dir: str,
    generators_txt: str,
    output_dir: str,
    holdout_generators=None,    # list of generators to exclude from training
    test_size: float = 0.20,     # fraction of TOTAL images to go into the test set
    random_state: int = 42
):
    """
    Create train / val / test CSVs with columns: filepath,label,generator

    • Splits the full dataset into 80% train+val and 20% test.
    • From the train+val pool, splits 80% train and 20% val.
    • Any image whose generator is in holdout_generators is prevented from entering the train set.
      (They may appear in val or test, depending on random split.)
    """

    rng = np.random.default_rng(random_state)

    # ─────────────────────────────────── 1. read mapping of fake images to generators
    fake_image_to_gen = {}
    with open(generators_txt, 'r') as f:
        for line in f:
            img, gen = line.strip().split()[:2]
            fake_image_to_gen[img] = gen

    # ─────────────────────────────────── 2. build fake / real DataFrames
    fake_df = pd.DataFrame([
        dict(filepath=str(Path(fake_images_dir) / fn),
             label="fake",
             generator=fake_image_to_gen.get(fn, "unknown"))
        for fn in os.listdir(fake_images_dir) if fn.endswith(".png")
    ])

    real_df = pd.DataFrame([
        dict(filepath=str(Path(real_images_dir) / fn),
             label="real",
             generator="real")
        for fn in os.listdir(real_images_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # ─────────────────────────────────── 3. combine
    all_df = pd.concat([fake_df, real_df], ignore_index=True)

    # ─────────────────────────────────── 4. separate holdout vs rest
    if holdout_generators:
        mask_hold = all_df["generator"].isin(holdout_generators)
        holdout_df = all_df[mask_hold].copy()
        rest_df    = all_df[~mask_hold].copy()
    else:
        holdout_df = pd.DataFrame(columns=all_df.columns)
        rest_df    = all_df.copy()

    total_images    = len(all_df)
    desired_test_sz = int(round(test_size * total_images))

    # ─────────────────────────────────── 5. build test set
    #   include all holdout images, plus a random sample from rest to reach desired size
    n_rest_needed = max(0, min(len(rest_df), desired_test_sz - len(holdout_df)))
    rest_test_df = rest_df.sample(n=n_rest_needed, random_state=random_state)
    test_df = pd.concat([holdout_df, rest_test_df], ignore_index=True) \
                .sample(frac=1, random_state=random_state) \
                .reset_index(drop=True)

    # ─────────────────────────────────── 6. build train+val pool
    train_val_df = rest_df.drop(index=rest_test_df.index)

    # ─────────────────────────────────── 7. split train+val into train / val (80% / 20%)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.20,
        random_state=random_state,
        shuffle=True
    )
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # ─────────────────────────────────── 8. write CSVs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(output_dir) / "train.csv", index=False)
    val_df.to_csv  (Path(output_dir) / "val.csv",   index=False)
    test_df.to_csv (Path(output_dir) / "test.csv",  index=False)

    print(
        f"Splits saved to '{output_dir}': "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
fake_images_dir = "data/fake_images"
real_images_dir = "data/real_images"
generators_txt = "data/fake_images/generators.txt"
output_dir = "data/splits"

holdout_generators = ["Flux_1", "Mobius", "SDXL_Turbo"]  

create_stratified_splits(
    fake_images_dir=fake_images_dir,
    real_images_dir=real_images_dir,
    generators_txt=generators_txt,
    output_dir=output_dir,
    holdout_generators=holdout_generators,
    test_size=0.15,
    random_state=17
)

