import pandas as pd
import numpy as np
import ast 

df = pd.read_csv("data/output_data.csv")

def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val #debugging

df["Q"] = df["Q"].apply(safe_eval)
df["Miller_Indices"] = df["Miller_Indices"].apply(safe_eval)
df["U"] = df["U"].apply(safe_eval)
df["Unit_Cell"] = df["Unit_Cell"].apply(safe_eval)

df["Q"] = df["Q"].apply(lambda x: np.array(x) if isinstance(x, list) else x)
df["Miller_Indices"] = df["Miller_Indices"].apply(lambda x: np.array(x) if isinstance(x, list) else x)
df["U"] = df["U"].apply(lambda x: np.array(x) if isinstance(x, list) else x)
df["Unit_Cell"] = df["Unit_Cell"].apply(lambda x: np.array(x) if isinstance(x, list) else x)

print(df.dtypes)
print(df.head())

np.save("data/output_data.npy", df.to_dict(orient="records"))
