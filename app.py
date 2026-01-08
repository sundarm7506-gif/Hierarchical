import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Hierarchical Clustering", layout="centered")

st.title("Hierarchical Clustering")
st.write("Large datasets are automatically sampled to avoid memory errors.")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
   
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    df_num = df.select_dtypes(include=["int64", "float64"])
    
    if df_num.shape[1] < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
        st.stop()

   
    st.sidebar.header("⚙️ Clustering Settings")

    sample_size = st.sidebar.slider(
        "Sample size for clustering",
        min_value=500,
        max_value=5000,
        value=3000,
        step=500
    )

    n_clusters = st.sidebar.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=4
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

   
    np.random.seed(42)
    sample_idx = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
    X_sample = X_scaled[sample_idx]
    df_sample = df.iloc[sample_idx].copy()

   
    st.subheader(" Dendrogram (Sampled Data)")

    fig, ax = plt.subplots(figsize=(10, 5))
    linked = linkage(X_sample, method="ward")

    dendrogram(
        linked,
        truncate_mode="lastp",
        p=30,
        ax=ax
    )

    ax.set_title("Hierarchical Dendrogram")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Distance")

    st.pyplot(fig)


    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward"
    )

    clusters = hc.fit_predict(X_sample)
    df_sample["Cluster"] = clusters

    st.subheader("Cluster Summary (Mean Values)")
    st.dataframe(df_sample.groupby("Cluster").mean())

   
    st.subheader("Cluster Visualization")

    x_col = df_num.columns[0]
    y_col = df_num.columns[1]

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(
        df_sample[x_col],
        df_sample[y_col],
        c=df_sample["Cluster"],
        cmap="viridis",
        s=20
    )

    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.set_title("Hierarchical Clustering (Sampled Data)")
    plt.colorbar(scatter, ax=ax2, label="Cluster")

    st.pyplot(fig2)

   
    st.subheader("⬇️ Download Clustered Data")
    st.download_button(
        "Download CSV",
        data=df_sample.to_csv(index=False),
        file_name="hierarchical_clusters.csv",
        mime="text/csv"
    )
