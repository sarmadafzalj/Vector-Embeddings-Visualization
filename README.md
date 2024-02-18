# Visualize the Vector Embeddings
Welcome to the Visualizing Vector Embeddings project! This project focuses on visualizing vector embeddings in a RAG (Retrieval-Augmented Generation) system using Python, Pinecone, and Plotly.

## Overview
Today, most people are experts in creating RAG systems with vector search, but have you ever wondered how it brings relevant data? Or what your question looks like in the vector space?

In this project, we create vector embeddings for a PDF file and load them into Pinecone. Then, using t-SNE (t-distributed Stochastic Neighbor Embedding), we reduce the vector dimensions from 1536D (for ada-02) to 3D for plotting and visualization.

## Features
Create Embeddings and upsert in Pinecone
Ask a question, create its embeddings using OpenAI
Fetch all vectors related to the question and calculate their similarity scores
Reduce dimensions to 3D using t-SNE
Color the three most similar vectors and your query differently from the rest
Plot the vector space using Plotly

## Setup
- Clone the repo and cd into **Vector-Embeddings-Visualization**
- Create Pinecone account and get the API Key, free tier is available
- Get OpenAI API key for embeddings
- Upload your data in the Data/ folder
- Copy the notebook and replace the API keys and run it

## Reach out to me
- <i>Author: <b>Sarmad Afzal</b></i>
- <i>Linkedin: https://www.linkedin.com/in/sarmadafzal/</i>
- <i>Github: https://github.com/sarmadafzalj</i>
- <i>Youtube: https://www.youtube.com/@sarmadafzalj</i>
---