import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = "sediment-multi-environment-learning-test-1"

loss_bound_df = pd.read_csv(f"./{folder}/loss_bound.log")
loss_h_df = pd.read_csv(f"./{folder}/loss_h.log")
loss_momentum_df = pd.read_csv(f"./{folder}/loss_momentum.log")
loss_sediment_df = pd.read_csv(f"./{folder}/loss_sediment.log")

df = pd.merge(loss_bound_df, loss_h_df, on="Index", how='inner')
df = pd.merge(df, loss_momentum_df, on="Index", how='inner')
df = pd.merge(df, loss_sediment_df, on="Index", how='inner')

df.plot(x='Index', y=["loss_h", "loss_momentum", "loss_bound", "loss_sediment"])

plt.show()