# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scipy.signal as ss
import tensorflow as tf
from plotly.subplots import make_subplots

from utils.data_reader import read_h5_data

input_features = ["f", "zeta", "pv"]

with open("GRI_species_order_lu13", "r") as f:
    labels13 = [a.rstrip() for a in f.readlines()]

with open("GRI_species_order", "r") as f:
    labels = [a.rstrip() for a in f.readlines()]
labels.append("T")
labels.append("PVs")

x, y, df, in_scaler, out_scaler = read_h5_data(
    # "./data/tables_of_fgm.h5",
    "./data/tables_of_fgm.parquet",
    input_features=input_features,
    labels=labels,
    i_scaler="no",
    o_scaler="cbrt_std",
)
df["zetaLevel"] = df.zeta.apply(str)
df_filtered = df.copy()

# %%


def wireframe_html(df, df_filtered, sp, filtered=False):
    df = df
    df_filtered = df_filtered
    wireframe = []

    def plot_line(x_grid, y_grid, z_grid, wireName, color):
        lines = []
        line_marker = dict(color=color, width=2)
        legendFlag = True

        # wireName = f"{sp}:z={zetaLevel}(filtered:{filtered})"

        for i, j, k in zip(x_grid, y_grid, z_grid):
            lines.append(
                go.Scatter3d(
                    x=i,
                    y=j,
                    z=k,
                    mode="lines",
                    line=line_marker,
                    legendgroup=wireName,
                    name=wireName,
                    showlegend=legendFlag,
                )
            )
            legendFlag = False
        for i, j, k in zip(x_grid.T, y_grid.T, z_grid.T):
            lines.append(
                go.Scatter3d(
                    x=i,
                    y=j,
                    z=k,
                    mode="lines",
                    line=line_marker,
                    # legendgroup=f"{sp}:z={zetaLevel}",
                    legendgroup=wireName,
                    showlegend=False,
                )
            )

        wireframe.append(lines)

    for zetaLevel in list(set(df.zetaLevel)):
        df_z0 = df[df.zetaLevel == zetaLevel]

        x_grid = df_z0.f.values.reshape(501, 501)
        y_grid = df_z0.pv.values.reshape(501, 501)
        z_grid = df_z0[sp].values.reshape(501, 501)
        # if (filtered == True) and zetaLevel not in ["0.0"]:
        if filtered == True:
            if zetaLevel in ["0.0", "0.11"]:
                z_grid_filtered = ss.savgol_filter(z_grid, 7, 4)
            else:
                z_grid_filtered = ss.savgol_filter(z_grid, 31, 4)
            # update filtered values
            df_filtered.loc[
                df_filtered.zetaLevel == zetaLevel, sp
            ] = z_grid_filtered.reshape(-1, 1)
            wireName = f"{sp}:z={zetaLevel}(filtered:{filtered})"
            plot_line(x_grid, y_grid, z_grid_filtered, wireName, color="#0066ff")

        wireName = f"{sp}:z={zetaLevel}"
        plot_line(x_grid, y_grid, z_grid, wireName, color="#ff0000")

    fig = go.Figure(data=[line for lines in wireframe for line in lines[::10]])
    # fig.show()
    if filtered == True:
        fig.write_html(f"./wireframe/{sp}_wire_filtered.html")
    else:
        fig.write_html(f"./wireframe/{sp}_wire.html")


# %%
for i in range(0, 3):
    print("i = ", i)
    df = df_filtered.copy()
    # labelList = labels13+['T','PVs']
    labelList = labels
    # labelList = ['O','PVs','HO2','CH3']
    for sp in enumerate(labelList):
        # wireframe_html(sp)
        print(f"{sp[0]+1}/{len(labelList)},{sp[1]}")
        wireframe_html(df=df, df_filtered=df_filtered, sp=sp[1], filtered=True)

    df_filtered.drop("zetaLevel", axis=1, inplace=True)
    df_filtered = df_filtered.clip(lower=0)

    # df_filtered["zetaLevel"]=df["zetaLevel"]
    df_filtered["zetaLevel"] = df_filtered.zeta.apply(str)
    df["zetaLevel"] = df.zeta.apply(str)
# df = df_filtered
# %%
df_filtered.to_parquet("df_filtered.parquet")

# %%
f_id = -1
zetaLevel = "0.44"
# for sp in labels13+['PVs']:
for sp in ["PVs", "O", "HO2", "CH3"]:
    # for sp in ["PVs", "CH2O","CH3","O"]:
    val_org = df[df.zetaLevel == zetaLevel][sp].values.reshape(501, 501)
    # val_filtered = ss.savgol_filter(val_org, 31, 4)
    val_filtered = df_filtered[df_filtered.zetaLevel == zetaLevel][sp].values.reshape(
        501, 501
    )

    plt.figure()
    plt.plot(val_org[f_id], "r-")
    plt.plot(val_filtered[f_id])
    plt.title(sp)
    plt.show()

# %%

# %%

# a = pd.read_hdf('df_test.h5')
dataFile = "./data/tables_of_fgm.h5"
a = pd.read_hdf(dataFile)
# %%
a.to_parquet("tables_of_fgm.parquet")
# a.to_hdf('df_test.h5',key='hdf')
# %%

b = pd.read_parquet("tables_of_fgm.parquet")
# b = pd.read_hdf('df_test.h5')
b.shape
# %%
dataFile.split(".")[-1] == "h5"

# %%
df_filtered.shape

# %%
df_filtered.drop("zetaLevel", axis=1, inplace=True)
df_filtered = df_filtered.clip(lower=0)
df_filtered.to_parquet("df_filtered.parquet")
# %%
df_filtered = pd.read_parquet("df_filtered.parquet")
df_filtered["zetaLevel"] = df["zetaLevel"]


with open("out_scaler.pkl", "rb") as f:
    out_scaler = pickle.load(f)

model = tf.keras.models.load_model("wudi_4x100.h5")

# %%
x = df[input_features].values
pred_org = model.predict(x, batch_size=1024)

# %%
out_sps = labels13 + ["T", "PVs"]
out_sps.remove("N2")


# %%
df_pred = pd.DataFrame(out_scaler.inverse_transform(pred_org), columns=out_sps)

df_model = pd.concat([df[input_features], df_pred], axis=1)
df_model["zetaLevel"] = df_model.zeta.apply(str)
# %%


# %%
px.scatter_3d(
    data_frame=df_model[df_model.zetaLevel == "0.0"].sample(n=10_000),
    x="f",
    y="pv",
    z="PVs",
)

# %%
