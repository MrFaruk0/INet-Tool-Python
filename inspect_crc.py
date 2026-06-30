import pandas as pd, numpy as np

for name in ["species_all", "pathway_all", "enzyme_all"]:
    path = fr"C:\Users\Faruk\Desktop\VSCode Projeler\INet-Tool-Python\CRC-Dataset\{name}.xlsx"
    df = pd.read_excel(path, index_col=0)
    print(f"=== {name} ===")
    print(f"  Shape: {df.shape}")
    print(f"  Index name: '{df.index.name}'")
    idx_vals = df.index.value_counts()
    print(f"  Index value counts: {dict(idx_vals)}")
    print(f"  No NaN: {df.isna().sum().sum() == 0}")
    print(f"  Value stats: min={df.values.min():.2f}, max={df.values.max():.2f}")
    # Check if data is counts or normalized
    print(f"  Mean per row range: {df.mean(axis=1).min():.2f} - {df.mean(axis=1).max():.2f}")
    print(f"  Std per row range: {df.std(axis=1).min():.2f} - {df.std(axis=1).max():.2f}")
    print(f"  Fraction of zeros: {(df.values == 0).mean():.3f}")
    print()
