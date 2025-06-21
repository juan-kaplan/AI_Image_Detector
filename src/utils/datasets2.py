import os
from pathlib import Path
import numpy as np
import pandas as pd

def create_stratified_splits(
    fake_images_dir: str,
    real_images_dir: str,
    generators_txt: str,
    output_dir: str,
    holdout_generators=None,    # list of generators to exclude from training
    test_size: float = 0.20,    # fraction of TOTAL images to go into the test set
    random_state: int = 42
):
    if holdout_generators is None:
        holdout_generators = []

    rng = np.random.default_rng(random_state)

    # map fake image filenames to their generator
    fake_image_to_gen = {}
    with open(generators_txt, 'r') as f:
        for line in f:
            img, gen = line.strip().split()[:2]
            fake_image_to_gen[img] = gen

    # build DataFrames
    fake_df = pd.DataFrame([
        dict(filepath=str(Path(fake_images_dir) / fn),
             label="fake",
             generator=fake_image_to_gen.get(fn, "unknown"))
        for fn in os.listdir(fake_images_dir)
        if fn.lower().endswith(".png")
    ])

    real_df = pd.DataFrame([
        dict(filepath=str(Path(real_images_dir) / fn),
             label="real",
             generator="real")
        for fn in os.listdir(real_images_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    total_n = 2 * len(fake_df)
    fake_holdout_df = fake_df[fake_df['generator'].isin(holdout_generators)]
    fake_df_train   = fake_df[~fake_df['generator'].isin(holdout_generators)]

    # 1) SAMPLE TEST
    test_n      = int(total_n * test_size)
    fake_test_n = test_n // 2
    real_test_n = test_n - fake_test_n

    if holdout_generators:
        test_fake = (
            fake_holdout_df
            .groupby('generator', group_keys=False)[['filepath','label','generator']]
            .apply(lambda x: x.sample(
                min(fake_test_n // len(holdout_generators), len(x)),
                random_state=random_state
            ))
        )
        # remove those exact rows from holdout pool
        remaining_fake_hold = fake_holdout_df.drop(test_fake.index, errors='ignore')
        test_fake = test_fake.reset_index(drop=True)
    else:
        test_fake = pd.DataFrame(columns=fake_df.columns)
        remaining_fake_hold = fake_holdout_df.copy()

    test_real      = real_df.sample(n=real_test_n, random_state=random_state)
    remaining_real = real_df.drop(test_real.index)

    # 2) SAMPLE VALIDATION
    val_n      = int(total_n * (1 - test_size) * test_size)
    fake_val_n = val_n // 2
    real_val_n = val_n - fake_val_n

    if holdout_generators:
        # 2a) grab what we can from holdout
        val_fake_holdout = (
            remaining_fake_hold
            .groupby('generator', group_keys=False)[['filepath','label','generator']]
            .apply(lambda x: x.sample(
                min(fake_val_n // len(holdout_generators), len(x)),
                random_state=random_state
            ))
        )
        # drop them from holdout pool
        remaining_fake_hold = remaining_fake_hold.drop(val_fake_holdout.index, errors='ignore')

        # 2b) if not enough holdouts, pull the rest from non-holdout pool
        if len(val_fake_holdout) < fake_val_n:
            needed = fake_val_n - len(val_fake_holdout)
            val_fake_extra = fake_df_train.sample(n=min(needed, len(fake_df_train)),
                                                  random_state=random_state)
            # remove these from train‐pool to avoid overlap
            fake_df_train = fake_df_train.drop(val_fake_extra.index)
            val_fake = pd.concat([val_fake_holdout, val_fake_extra])
        else:
            val_fake = val_fake_holdout

        val_fake = val_fake.reset_index(drop=True)
    else:
        val_fake = pd.DataFrame(columns=fake_df.columns)

    val_real      = remaining_real.sample(n=real_val_n, random_state=random_state)
    remaining_real = remaining_real.drop(val_real.index)

    # 3) SAMPLE TRAIN
    used_n       = len(test_fake) + len(test_real) + len(val_fake) + len(val_real)
    train_n      = total_n - used_n
    fake_train_n = train_n // 2
    real_train_n = train_n - fake_train_n

    train_fake = fake_df_train.sample(n=fake_train_n, random_state=random_state)
    train_real = remaining_real.sample(n=real_train_n, random_state=random_state)

    # 4) ASSEMBLE & SAVE
    train_df = pd.concat([train_fake, train_real])\
                 .sample(frac=1, random_state=random_state)\
                 .reset_index(drop=True)
    val_df   = pd.concat([val_fake,   val_real])\
                 .sample(frac=1, random_state=random_state)\
                 .reset_index(drop=True)
    test_df  = pd.concat([test_fake,  test_real])\
                 .sample(frac=1, random_state=random_state)\
                 .reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(Path(output_dir) / 'train.csv', index=False)
    val_df.to_csv(  Path(output_dir) / 'val.csv',   index=False)
    test_df.to_csv( Path(output_dir) / 'test.csv',  index=False)

    # report breakdown
    for name, df in (("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)):
        c = df['label'].value_counts().to_dict()
        print(f"{name}: total={len(df):5d}, fake={c.get('fake', 0):5d}, real={c.get('real', 0):5d}")

fake_images_dir = "data/fake_images"
real_images_dir = "data/real_images"
generators_txt = "data/fake_images/generators.txt"
output_dir = "data/splits"

holdout_generators = ["Flux_1", "Mobius", "SDXL_Turbo", 'Lumina', 'PixArt_Alpha']  

create_stratified_splits(
    fake_images_dir=fake_images_dir,
    real_images_dir=real_images_dir,
    generators_txt=generators_txt,
    output_dir=output_dir,
    holdout_generators=holdout_generators,
    test_size=0.15,
    random_state=17
)



