
import pandas as pd

# Load processed dataset
df = pd.read_csv("dataset/processed_dataset.csv")

print("=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

# Dataset information
print("\nDataset Shape:", df.shape)

print("\nTotal Records:", len(df))

print("Total Features:", len(df.columns) - 1)

# Traffic distribution
print("\nTraffic Category Distribution:")

print(df["Label"].value_counts().to_string())

# Check class percentages
print("\nTraffic Category Percentages:")

print(
    (df["Label"].value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)

# Check missing values
print("\nMissing Values:", df.isnull().sum().sum())

# Check duplicates
print("Duplicate Records:", df.duplicated().sum())

print("\nDATASET SUMMARY COMPLETED!")
