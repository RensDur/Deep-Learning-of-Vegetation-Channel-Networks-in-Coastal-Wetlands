import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = "Saltmarsh test 1"

loss_bound_df = pd.read_csv(f"./{folder}/loss_bound.log")
loss_h_df = pd.read_csv(f"./{folder}/loss_h.log")
loss_momentum_df = pd.read_csv(f"./{folder}/loss_momentum.log")
loss_sediment_df = pd.read_csv(f"./{folder}/loss_sediment.log")
loss_vegetation_df = pd.read_csv(f"./{folder}/loss_vegetation.log")

df = pd.merge(loss_bound_df, loss_h_df, on="Index", how='inner')
df = pd.merge(df, loss_momentum_df, on="Index", how='inner')
df = pd.merge(df, loss_sediment_df, on="Index", how='inner')
df = pd.merge(df, loss_vegetation_df, on="Index", how='inner')

df.plot(x='Index', y=["loss_h", "loss_momentum", "loss_sediment", "loss_vegetation", "loss_bound"])

plt.show()