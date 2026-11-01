
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

DATASET_DIR = Path("dataset")
OUTPUT_FILE = DATASET_DIR / "processed_dataset.csv"

CHUNK_SIZE = 100000
MAX_PER_CLASS = 20000
PER_CHUNK_SAMPLE = 2000

print("=" * 65)
print("NETWORK INTRUSION DETECTION")
print("DATA PREPROCESSING STARTED")
print("=" * 65)


# ==========================================
# 2. FIND ORIGINAL CSV FILES
# ==========================================

csv_files = sorted(
    file for file in DATASET_DIR.glob("*.csv")
    if file.name != OUTPUT_FILE.name
)

if not csv_files:
    raise FileNotFoundError(
        "No original CSV files found in the dataset folder."
    )

print("\nTotal Original CSV Files:", len(csv_files))


# ==========================================
# 3. PROCESS DATA IN CHUNKS
# ==========================================

sampled_chunks = []

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    try:

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
            low_memory=False
        ):

            # Clean column names
            chunk.columns = chunk.columns.str.strip()

            # Check Label column
            if "Label" not in chunk.columns:
                print("Warning: Label column missing.")
                continue

            # Clean traffic labels
            chunk["Label"] = (
                chunk["Label"]
                .astype(str)
                .str.strip()
            )

            # Remove empty or invalid labels
            chunk = chunk[
                chunk["Label"].notna()
                & (chunk["Label"] != "")
                & (chunk["Label"].str.lower() != "nan")
            ]

            if chunk.empty:
                continue

            # ==================================
            # 4. REMOVE IDENTIFIER COLUMNS
            # ==================================

            identifier_columns = [
                "Flow ID",
                "Source IP",
                "Destination IP",
                "Timestamp"
            ]

            chunk.drop(
                columns=identifier_columns,
                errors="ignore",
                inplace=True
            )

            # ==================================
            # 5. CONVERT FEATURES TO NUMERIC
            # ==================================

            feature_columns = [
                col for col in chunk.columns
                if col != "Label"
            ]

            for col in feature_columns:

                chunk[col] = pd.to_numeric(
                    chunk[col],
                    errors="coerce"
                )

            # ==================================
            # 6. HANDLE INFINITE VALUES
            # ==================================

            chunk.replace(
                [np.inf, -np.inf],
                np.nan,
                inplace=True
            )

            # Remove rows containing missing values
            chunk.dropna(
                subset=feature_columns + ["Label"],
                inplace=True
            )

            if chunk.empty:
                continue

            # ==================================
            # 7. SAMPLE TRAFFIC CATEGORIES
            # ==================================

            for label, group in chunk.groupby(
                "Label",
                sort=False
            ):

                sample_size = min(
                    len(group),
                    PER_CHUNK_SAMPLE
                )

                sampled_group = group.sample(
                    n=sample_size,
                    random_state=42
                )

                sampled_chunks.append(sampled_group)

        print("Completed:", file.name)

    except Exception as e:

        print("\nError processing:", file.name)
        print("Error details:", e)


# ==========================================
# 8. COMBINE SAMPLED DATA
# ==========================================

if not sampled_chunks:

    raise ValueError(
        "No valid records collected. "
        "Please check your original CSV files."
    )

print("\nCombining sampled records...")

data = pd.concat(
    sampled_chunks,
    ignore_index=True
)

print("Combined Dataset Shape:", data.shape)


# ==========================================
# 9. REMOVE DUPLICATE RECORDS
# ==========================================

before_duplicates = len(data)

data.drop_duplicates(
    inplace=True
)

after_duplicates = len(data)

print("\nDuplicate Records Removed:",
      before_duplicates - after_duplicates)


# ==========================================
# 10. LIMIT EACH CLASS
# ==========================================

print("\nApplying per-class sampling...")

balanced_parts = []

for label, group in data.groupby(
    "Label",
    sort=False
):

    sample_size = min(
        len(group),
        MAX_PER_CLASS
    )

    sampled_group = group.sample(
        n=sample_size,
        random_state=42
    )

    balanced_parts.append(sampled_group)

# Combine all sampled classes
balanced_data = pd.concat(
    balanced_parts,
    ignore_index=True
)


# ==========================================
# 11. SHUFFLE DATASET
# ==========================================

balanced_data = balanced_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 12. SAVE PROCESSED DATASET
# ==========================================

balanced_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 13. FINAL DATASET SUMMARY
# ==========================================

print("\n" + "=" * 65)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFinal Dataset Shape:")
print(balanced_data.shape)

print("\nTotal Records:")
print(len(balanced_data))

print("\nTotal Features:")
print(len(balanced_data.columns) - 1)

print("\nTraffic Category Distribution:")

print(
    balanced_data["Label"]
    .value_counts()
    .to_string()
)

print("\nMissing Values:")
print(balanced_data.isnull().sum().sum())

print("\nInfinite Values:")
print(
    np.isinf(
        balanced_data.select_dtypes(
            include=np.number
        )
    ).sum().sum()
)

print("\nSaved Dataset Location:")
print(OUTPUT_FILE)

print("\n" + "=" * 65)
print("DATA PREPROCESSING FINISHED!")
print("=" * 65)
