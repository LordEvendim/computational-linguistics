from datasets import load_dataset

ds = load_dataset("jziebura/polish_youth_slang_classification")

df_train = ds["train"].to_polars()
df_test = ds["test"].to_polars()
df_val = ds["validation"].to_polars()
