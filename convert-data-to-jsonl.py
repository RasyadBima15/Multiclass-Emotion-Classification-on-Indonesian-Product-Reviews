import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset_augmented.csv")

# Pastikan tidak ada nilai kosong
df = df.dropna(subset=["Customer Review", "Emotion"])

# -------------------------
# SPLIT DATASET
# -------------------------

# --- Step 1: Split Train 80% dan Temp 20% ---
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Emotion"],
    random_state=42
)

# --- Step 2: Split Temp 20% menjadi Val 10% + Test 10% ---
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["Emotion"],
    random_state=42
)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

# print total jumlah per kelas
print("\nJumlah per kelas di Train:")
print(train_df["Emotion"].value_counts())
print("\nJumlah per kelas di Validation:")
print(val_df["Emotion"].value_counts())
print("\nJumlah per kelas di Test:")
print(test_df["Emotion"].value_counts())

test_df.to_csv(
    "test_dataset_new.csv",
    index=False,
    encoding="utf-8"
)

# -------------------------
# FUNCTION TO SAVE JSONL
# -------------------------

def save_jsonl(dataframe, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Kamu adalah model klasifikasi emosi ulasan e-commerce."
                    },
                    {
                        "role": "user",
                        "content": str(row["Customer Review"])
                    },
                    {
                        "role": "assistant",
                        "content": str(row["Emotion"])
                    }
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

# # # -------------------------
# # # SAVE FILES
# # # -------------------------

save_jsonl(train_df, "train_new.jsonl")
save_jsonl(val_df, "validation_new.jsonl")