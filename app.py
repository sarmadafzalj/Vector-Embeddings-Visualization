from pinecone import Pinecone
import plotly.graph_objs as go
pc = Pinecone(api_key='2c9b3c26-9ac4-459b-8d1a-3c3fe6cc97b7')
index = pc.Index('mvpindex')

pc.describe_index('mvpindex')

from openai import OpenAI
client = OpenAI(api_key="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")

response = client.embeddings.create(
    input="What is peer evaluation?",
    model="text-embedding-ada-002"
)

query_emb = response.data[0].embedding
#print(query_emb)

search = index.query(
  vector=query_emb,
  top_k=900,
  include_values=True,
  namespace = "HuzefaCourse"
)

embs = []
for i in range(0,900):
    embs.append(search.matches[i].values)

embs.insert(0,query_emb)

n_components = 3
import numpy as np
from sklearn.manifold import TSNE

embs = np.array(embs)

tsne = TSNE(n_components=n_components, random_state=42, perplexity=5)
reduced_vectors = tsne.fit_transform(embs)

from plotly.subplots import make_subplots

# Create a 3D scatter plot
scatter_plot = go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=6, color='blue', opacity=0.8),
    text=[f"Point {i}" for i in range(len(reduced_vectors))]
)

# Highlight the first point with a different color
highlighted_point = go.Scatter3d(
    x=[reduced_vectors[0, 0]],
    y=[reduced_vectors[0, 1]],
    z=[reduced_vectors[0, 2]],
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.8),
    text=[f"Point 0"]
)

# Create the layout for the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='t-SNE Component 1'),
        yaxis=dict(title='t-SNE Component 2'),
        zaxis=dict(title='t-SNE Component 3'),
    ),
    title=f'3D Representation after t-SNE (Perplexity=5)'
)

# Enable interactive widget mode for Plotly in Jupyter Notebook
#%matplotlib widget

# Create the Figure using make_subplots for compatibility with interactive widgets
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# Add the scatter plots to the Figure
fig.add_trace(scatter_plot)
fig.add_trace(highlighted_point)

# Update the layout
fig.update_layout(layout)

# Show the interactive plot
fig.show()