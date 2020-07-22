#%%
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from utils.data_reader import read_h5_data

input_features = ["f", "zeta", "pv"]

with open("GRI_species_order", "r") as f:
    labels = [a.rstrip() for a in f.readlines()]
labels.append("T")
labels.append("PVs")

x, y, df, in_scaler, out_scaler = read_h5_data(
    "./data/tables_of_fgm.h5",
    input_features=input_features,
    labels=labels,
    i_scaler="no",
    o_scaler="cbrt_std",
)
df["zetaLevel"] = df.zeta.apply(str)


#%%
def wireframe_html(sp, filtered=False):

    wireframe = []

    for zetaLevel in list(set(df.zetaLevel)):
        df_z0 = df[df.zetaLevel == zetaLevel]

        x_grid = df_z0.f.values.reshape(501, 501)
        y_grid = df_z0.pv.values.reshape(501, 501)
        z_grid = df_z0[sp].values.reshape(501, 501)
        if filtered == True:
            z_grid = ss.savgol_filter(z_grid, 51, 4)

        #%%
        lines = []
        line_marker = dict(color="#0066FF", width=2)
        # line_marker = dict(width=2)
        legendFlag = True
        for i, j, k in zip(x_grid, y_grid, z_grid):
            lines.append(
                go.Scatter3d(
                    x=i,
                    y=j,
                    z=k,
                    mode="lines",
                    line=line_marker,
                    legendgroup=f"{sp}:z={zetaLevel}(filtered:{filtered})",
                    name=f"{sp}:z={zetaLevel}(filtered:{filtered})",
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
                    legendgroup=f"{sp}:z={zetaLevel}(filtered:{filtered})",
                    showlegend=False,
                )
            )
        wireframe.append(lines)

    #%%
    fig = go.Figure(data=[line for lines in wireframe for line in lines[::10]])
    fig.show()
    #%%

    # if filtered == True:
    #     fig.write_html(f"./wireframe/{sp}_wire_filtered.html")
    # else:
    #     fig.write_html(f"./wireframe/{sp}_wire.html")


#%%
# for sp in labels:
#     wireframe_html(sp)


# %%
import matplotlib.pyplot as plt

import scipy.signal as ss

# %%
df_tmp = df[df.zetaLevel == "0.33"]
df_filtered = df_tmp.copy()
# %%
# sp = "PVs"
# for sp in labels:
for sp in ["CH3O", "PVs", "T"]:
    val_org = df_tmp[sp].values.reshape(501, 501)
    val_filtered = ss.savgol_filter(val_org, 51, 4)

    df_filtered[sp] = val_filtered.reshape(-1, 1)

    f_id = -20
    plt.figure()
    plt.plot(val_org[f_id], "r:")
    plt.plot(val_filtered[f_id])
    plt.title(sp)
    plt.show()


# %%
wireframe_html("PVs")

# %%
wireframe_html("PVs", filtered=True)


#%%
# %%

# %%
px.scatter_3d(data_frame=df_tmp, x="f", y="pv", z=sp)

# %%
px.scatter_3d(data_frame=df_filtered, x="f", y="pv", z=sp)

# %%
