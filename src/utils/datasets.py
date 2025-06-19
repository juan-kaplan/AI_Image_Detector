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
    holdout_generators=None,         # list of 3 generators, or None ⇒ pick
    test_size: float = 0.15,         # fraction of TOTAL images we want in test
    random_state: int = 42
):
    """
    Create train / val / test CSVs with columns: filepath,label,generator

    • Every image whose generator is in *holdout_generators* → test
    • If that is < `test_size` of all data, add stratified extra fake images
    • Same number of real images is sampled for test (or all that exist)
    • Remaining fakes: 80 % train, 20 % val (stratified by generator)
    • Remaining reals: 80 % train, 20 % val (random)
    """

    rng = np.random.default_rng(random_state)

    # ─────────────────────────────────── 1. read mapping
    fake_image_to_gen = {}
    with open(generators_txt) as f:
        for line in f:
            img, gen = line.strip().split()[:2]
            fake_image_to_gen[img] = gen

    # ─────────────────────────────────── 2. build fake / real DataFrames
    fake_df = pd.DataFrame([
        dict(filepath=str(Path(fake_images_dir) / fn),
             label="fake",
             generator=fake_image_to_gen[fn])
        for fn in os.listdir(fake_images_dir) if fn.endswith(".png")
    ])

    real_df = pd.DataFrame([
        dict(filepath=str(Path(real_images_dir) / fn),
             label="real",
             generator="real")
        for fn in os.listdir(real_images_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # ─────────────────────────────────── 3. select hold-out generators
    all_gens = sorted(fake_df["generator"].unique())
    if holdout_generators is None:
        holdout_generators = rng.choice(all_gens, size=3, replace=False).tolist()
    else:
        if len(holdout_generators) != 3:
            raise ValueError("Exactly three hold-out generators are required.")
        unknown = set(holdout_generators) - set(all_gens)
        if unknown:
            raise ValueError(f"Unknown generators: {', '.join(unknown)}")

    # ─────────────────────────────────── 4. build initial test split
    fake_test_df = fake_df[fake_df["generator"].isin(holdout_generators)].copy()
    fake_rem_df  = fake_df[~fake_df["generator"].isin(holdout_generators)].copy()

    total_images     = len(fake_df) + len(real_df)
    desired_test_sz  = int(round(test_size * total_images))

    # helper: stratified sample of extra fakes if test is too small
    def _add_extra_fake(num_needed):
        if num_needed <= 0:
            return pd.DataFrame(columns=fake_df.columns)
        frac = num_needed / len(fake_rem_df)
        extra, remainder = train_test_split(
            fake_rem_df,
            test_size=frac,
            stratify=fake_rem_df["generator"],
            random_state=random_state,
        )
        fake_rem_df[:] = remainder  # mutate outer variable
        return extra

    shortfall = desired_test_sz - len(fake_test_df)
    if shortfall > 0:
        fake_test_df = pd.concat([fake_test_df, _add_extra_fake(shortfall // 2)])

    # ─────────────────────────────────── 5. match real images in test
    real_test_sz = min(len(real_df), len(fake_test_df))
    real_test_df, real_rem_df = train_test_split(
        real_df, test_size=real_test_sz, random_state=random_state, shuffle=True
    )

    # ─────────────────────────────────── 6. 80 % / 20 % split of leftovers
    fake_train_df, fake_val_df = train_test_split(
        fake_rem_df,
        test_size=0.20,
        stratify=fake_rem_df["generator"],
        random_state=random_state,
    )
    real_train_df, real_val_df = train_test_split(
        real_rem_df,
        test_size=0.20,
        random_state=random_state,
    )

    # ─────────────────────────────────── 7. combine + shuffle
    train_df = pd.concat([fake_train_df, real_train_df]).sample(
        frac=1, random_state=random_state).reset_index(drop=True)
    val_df   = pd.concat([fake_val_df,   real_val_df]).sample(
        frac=1, random_state=random_state).reset_index(drop=True)
    test_df  = pd.concat([fake_test_df,  real_test_df]).sample(
        frac=1, random_state=random_state).reset_index(drop=True)

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
