import pandas as pd, json
import umap.umap_ as umap, numpy as np, plotly.express as px


with open("data/embeddings.json") as f:
    df_embeddings = json.load(f)
with open("data/sequences.json", encoding="utf8") as f:
    df_sequences = json.load(f)


print("mapping")
points = umap.UMAP(n_components=2).fit_transform(np.array(list(df_embeddings.values()))).tolist()
print("mapped")
case_ids = list(df_embeddings.keys())
df_points = {}
for i in range(len(case_ids)):
    df_points[case_ids[i]] = points[i]
case_ids = []
sequences = [] 
embeddings_x = []
embeddings_y = []

print("extracting data")
for case_id in df_embeddings.keys():
    case_ids.append(case_id)
    embeddings_x.append(df_points[case_id][0])
    embeddings_y.append(df_points[case_id][1])
    sequences.append(df_sequences[case_id])
    

df = pd.DataFrame({"id":case_ids, "embedding-x":embeddings_x, "embedding-y":embeddings_y, "sequences":sequences})
print("plotting")
fig = px.scatter(df, x='embedding-x', y='embedding-y', custom_data=["sequences"])
fig.update_traces(hovertemplate="<br>".join(["%{customdata[0]}"]))
fig.show()