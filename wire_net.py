#%%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model

from utils.customObjects import SGDRScheduler, coeff_r2
from utils.data_reader import read_h5_data
from utils.resBlock import res_block, res_block_org
from utils.writeANNProperties import writeANNProperties

##########################
class lr_log(tf.keras.callbacks.Callback):
    # def on_train_batch_end(self, batch, log=None):
    # print("wudi")

    # def on_batch_end(self, batch, logs):
    # logs.update({"lr": 199})
    # print("lr:", self.model.optimizer._decayed_lr("float32").numpy())

    def on_epoch_end(self, batch, log={}):
        # def on_epoch_begin(self, batch, log={}):
        log.update({"lr": self.model.optimizer._decayed_lr("float32").numpy()})
        print(
            "lr_decay:", self.model.optimizer._decayed_lr("float32").numpy(),
        )


#%%
# Parameters
n_neuron = 100
branches = 3
scale = 3
batch_size = 256
epochs = 2000
vsplit = 0.1
batch_norm = False

# define the type of scaler: MinMax or Standard
scaler = "Standard"  # 'Standard' 'MinMax'

##########################
# DO NOT CHANGE THIS ORDER!!
input_features = ["f", "zeta", "pv"]

labels = []
with open("GRI_species_order_lu13", "r") as f:
    labels = [a.rstrip() for a in f.readlines()]

# append other fields: heatrelease,  T, PVs
# labels.append('heatRelease')
labels.append("T")
labels.append("PVs")
labels.remove("N2")

# # tabulate psi, mu, alpha
# labels.append('psi')
# labels.append('mu')
# labels.append('alpha')

# read in the data
X, y, df, in_scaler, out_scaler = read_h5_data(
    # "./data/tables_of_fgm.h5",
    "./data/df_filtered_3.parquet",
    input_features=input_features,
    labels=labels,
    i_scaler="no",
    o_scaler="cbrt_std",
)

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

# %%
print("set up ANN")

# ANN parameters
dim_input = X_train.shape[1]
dim_label = y_train.shape[1]


# This returns a tensor
inputs = Input(shape=(dim_input,))  # ,name='input_1')

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation="relu")(inputs)

x = res_block_org(x, n_neuron, stage=1, block="a", bn=batch_norm)
x = res_block_org(x, n_neuron, stage=1, block="b", bn=batch_norm)
x = res_block_org(x, n_neuron, stage=1, block="c", bn=batch_norm)
x = res_block_org(x, n_neuron, stage=1, block="d", bn=batch_norm)

x = Dense(100, activation="relu")(x)

predictions = Dense(dim_label, activation="linear")(x)

model = Model(inputs=inputs, outputs=predictions)

# get the model summary
model.summary()

# WARM RESTART
batch_size_list = [
    # batch_size,
    batch_size * 64,
    # batch_size * 32,
    # batch_size * 8,
    # batch_size * 4,
]

for this_batch in batch_size_list:
    # checkpoint (save the best model based validate loss)
    filepath = "./tmp/weights.best.cntk.hdf5"

    # check if there are weights
    if os.path.isfile(filepath):
        print("read model")
        model.load_weights(filepath)

    checkpoint = ModelCheckpoint(
        filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        period=10,
    )

    base = 8
    _t_mul = 2
    epochs = sum([base * _t_mul ** i for i in range(3)])

    learning_rate = 1e-4
    first_decay_steps = base * np.ceil(X_train.shape[0] * (1 - vsplit) / this_batch)
    lr_decayed = tf.keras.experimental.CosineDecayRestarts(
        learning_rate, first_decay_steps, t_mul=_t_mul
    )
    # lr_decayed.alpha = 0.01
    lr_decayed._m_mul = 0.8

    model.compile(
        loss="mse",
        # optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decayed),
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_decayed, momentum=0.9),
        # optimizer=tf.keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    # callbacks_list = [checkpoint, schedule, lr_log()]
    callbacks_list = [checkpoint, lr_log()]

    # fit the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=this_batch,
        validation_split=vsplit,
        verbose=2,
        callbacks=callbacks_list,
        shuffle=True,
    )

plt.plot(history.history["lr"])
#%%


# # %%
predict_val = model.predict(X_test)

X_test_df = pd.DataFrame(in_scaler.inverse_transform(X_test), columns=input_features)
y_test_df = pd.DataFrame(out_scaler.inverse_transform(y_test), columns=labels)

sp = "PVs"

# loss
fig = plt.figure()
plt.semilogy(history.history["loss"])
if vsplit:
    plt.semilogy(history.history["val_loss"])
plt.title("mse")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.savefig("./exported/Loss_%s_%s_%i.eps" % (sp, scaler, n_neuron), format="eps")
plt.show(block=False)

predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)

plt.figure()
plt.title("Error of %s " % sp)
plt.plot((y_test_df[sp] - predict_df[sp]) / y_test_df[sp])
plt.title(sp)
plt.savefig("./exported/Error_%s_%s_%i.eps" % (sp, scaler, n_neuron), format="eps")
plt.show(block=False)

plt.figure()
plt.scatter(predict_df[sp], y_test_df[sp], s=1)
plt.title("R2 for " + sp)
plt.savefig("./exported/R2_%s_%s_%i.eps" % (sp, scaler, n_neuron), format="eps")
plt.show(block=False)

for sp in labels:
    print(sp, r2_score(predict_df[sp], y_test_df[sp]))

#%%
a = (y_test_df[sp] - predict_df[sp]) / y_test_df[sp]
test_data = pd.concat([X_test_df, y_test_df], axis=1)
pred_data = pd.concat([X_test_df, predict_df], axis=1)

test_data.to_hdf("sim_check.H5", key="test")
pred_data.to_hdf("sim_check.H5", key="pred")

# Save model
sess = K.get_session()
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, "./exported/my_model")
model.save("FPV_ANN_tabulated_%s.H5" % scaler)

# write the OpenFOAM ANNProperties file
writeANNProperties(in_scaler, out_scaler, scaler)

# Convert the model to
# run -i k2tf.py --input_model='FPV_ANN_tabulated_Standard.H5' --output_model='exported/FPV_ANN_tabulated_Standard.pb'

#%%
model.save("wudi_4x100.h5")

# %%

# %%
a = pd.read_parquet("./data/df_filtered.parquet")


# %%
a.drop("zetaLevel", axis=1, inplace=True)

# %%
a.to_parquet("./data/df_filtered_3.parquet")


# %%
with open("out_scaler.pkl", "wb") as f:
    pickle.dump(out_scaler, f)

# %%
with open("out_scaler.pkl", "rb") as f:
    a = pickle.load(f)


# %%


# %%
a.inverse_transform(y)


learning_rate = 1e-3
global_step = 1000
first_decay_steps = 1000
# lr_decayed = tf.keras.experimental.cosine_decay_restarts(learning_rate, global_step,
#                                    first_decay_steps)
lr_decayed = tf.keras.experimental.CosineDecayRestarts(
    learning_rate, global_step, first_decay_steps
)

# %%
print(lr_decayed)

# %%
lr_decayed.numpy()
# %%


# %%
