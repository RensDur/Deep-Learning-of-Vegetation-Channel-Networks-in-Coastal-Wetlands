import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


loss_bound_df = pd.read_csv("./loss_bound.log")
loss_h_df = pd.read_csv("./loss_h.log")
loss_momentum_df = pd.read_csv("./loss_momentum.log")

df = pd.merge(loss_bound_df, loss_h_df, on="Index", how='inner')
df = pd.merge(df, loss_momentum_df, on="Index", how='inner')

df.plot(x='Index', y=["loss_h", "loss_momentum", "loss_bound"])

plt.show()