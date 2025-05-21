import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")
st.title("Customer Segmentation Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("your_clustered_data.csv")  
    return df

df = load_data()

show_raw = st.checkbox("Show raw data", False)

cluster_features = ['Age', 'Income', 'Total_Spent', 'Visit_Purchase_Ratio_Capped']
cluster_summary = df.groupby('Cluster')[cluster_features].mean()
st.dataframe(cluster_summary.style.format("{:.2f}"))

scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(cluster_summary),
    columns=cluster_summary.columns,
    index=cluster_summary.index
)


fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=plot_data, x='Feature', y='Value', hue='Cluster', ax=ax)
plt.title("Normalized Feature Comparison per Cluster")
plt.xticks(rotation=45)
st.pyplot(fig)


cluster_counts = df['Cluster'].value_counts().sort_index()
st.bar_chart(cluster_counts)
