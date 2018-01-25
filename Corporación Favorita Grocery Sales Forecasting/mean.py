import itertools

import pandas as pd
import numpy as np

df_train2 = pd.read_csv(
    'input/train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': str},
    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0},
    skiprows=range(1, 124035460)
)

# log transform
df_train2["unit_sales"] = df_train2["unit_sales"].apply(np.log1p)

# Fill gaps in dates
# Improved with the suggestion from Paulo Pinto
u_dates = df_train2.date.unique()
u_stores = df_train2.store_nbr.unique()
u_items = df_train2.item_nbr.unique()
df_train2.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
df_train2 = df_train2.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=["date", "store_nbr", "item_nbr"]
    )
)

print("Nulls in Train columns: {0} => {1}".format(df_train.columns.values, df_train.isnull().any().values))

# Fill NAs
df_train2.loc[:, "unit_sales"].fillna(0, inplace=True)
# Assume missing entris imply no promotion
df_train2.loc[:, "onpromotion"].fillna("False", inplace=True)

df_train2.reset_index(inplace=True)

# Calculate means 
df_train2 = df_train2.groupby(
    ['item_nbr', 'store_nbr', 'onpromotion']
)['unit_sales'].mean().to_frame('unit_sales')
# Inverse transform
df_train2["unit_sales"] = df_train2["unit_sales"].apply(np.expm1)

# Create submission
pd.read_csv(
    "input/test.csv", usecols=[0, 2, 3, 4], dtype={'onpromotion': str}
).set_index(
    ['item_nbr', 'store_nbr', 'onpromotion']
).join(
    df_train2, how='left'
).fillna(0).to_csv(
    'mean.csv.gz', float_format='%.2f', index=None, compression="gzip"
)