import random
import pandas as pd

df= pd.read_csv("data/manifest_raw.csv")
df= df[df.n_faces>=16].copy()

df["group"]= df["stem"].str.split("_").str[0]
groups= sorted(df["group"].unique())
random.Random(42).shuffle(groups)

n=len(groups)
split_of={}
for i,g in enumerate(groups):
    split_of[g]= "train" if i<0.7*n else "val" if i<0.85*n else "test"
df["split"]= df["group"].map(split_of)

df.to_csv("data/manifest.csv", index=False)
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
print(df.groupby("split").n_faces.sum())