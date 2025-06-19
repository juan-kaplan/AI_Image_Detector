import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_splits(
    fake_images_dir,
    real_images_dir,
    generators_txt,
    output_dir,
    test_size=0.15,
    val_size=0.15,
    random_state=42
):
    """
    Create stratified train/val/test splits for fake and real images.
    Splits are stratified by generator for fake images.
    Writes CSVs with columns: filepath,label,generator
    """
    # Read fake image generator mapping
    fake_image_to_gen = {}
    with open(generators_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                fake_image_to_gen[parts[0]] = parts[1]

    # Collect fake images
    fake_images = [f for f in os.listdir(fake_images_dir) if f.endswith('.png')]
    fake_data = [
        {
            'filepath': os.path.join(fake_images_dir, fname),
            'label': 'fake',
            'generator': fake_image_to_gen.get(fname, 'unknown')
        }
        for fname in fake_images
    ]
    fake_df = pd.DataFrame(fake_data)

    # Collect real images
    real_images = [f for f in os.listdir(real_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    real_data = [
        {
            'filepath': os.path.join(real_images_dir, fname),
            'label': 'real',
            'generator': 'real'
        }
        for fname in real_images
    ]
    real_df = pd.DataFrame(real_data)

    # Stratified split for fake images
    fake_trainval, fake_test = train_test_split(
        fake_df, test_size=test_size, stratify=fake_df['generator'], random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    fake_train, fake_val = train_test_split(
        fake_trainval, test_size=val_ratio, stratify=fake_trainval['generator'], random_state=random_state
    )

    # Random split for real images (to match fake split sizes)
    real_trainval, real_test = train_test_split(
        real_df, test_size=len(fake_test), random_state=random_state
    )
    real_train, real_val = train_test_split(
        real_trainval, test_size=len(fake_val), random_state=random_state
    )

    # Combine splits
    train_df = pd.concat([fake_train, real_train]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat([fake_val, real_val]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([fake_test, real_test]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Write CSVs
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"CSVs written to {output_dir}")

