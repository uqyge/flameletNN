#%%
import random

import matplotlib.pyplot as plt
import plotly.express as px

from utils.data_reader import read_h5_data

input_features = ["f", "zeta", "pv"]

with open("GRI_species_order", "r") as f:
    labels = [a.rstrip() for a in f.readlines()]

# append other fields: heatrelease,  T, PVs
# labels.append('heatRelease')
labels.append("T")
labels.append("PVs")

#%%

x, y, df, in_scaler, out_scaler = read_h5_data(
    "./data/tables_of_fgm.h5",
    input_features=input_features,
    labels=labels,
    i_scaler="no",
    o_scaler="cbrt_std",
)


# %%
zetaLevel = list(set(df.zeta))
df_sample = df[df.zeta == zetaLevel[0]].sample(n=5_000)

sp = "T"
px.scatter_3d(
    data_frame=df_sample, x="f", y="pv", z=sp, color=sp, width=800, height=800
)

# %%
for sp in random.choices(labels, k=3):
    # for sp in ['T']:
    print(sp)
    px.scatter_3d(
        data_frame=df_sample, x="f", y="pv", z=sp, color=sp, width=800, height=800
    ).show()

# %%
