import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")
st.title("Customer Segmentation Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("your_clustered_data.csv")  # Make sure path is correct!
    return df

df = load_data()

# Show raw data toggle
show_raw = st.checkbox("Show raw data", False)
if show_raw:
    st.dataframe(df)

# Features to analyze
cluster_features = ['Age', 'Income', 'Total_Spent', 'Visit_Purchase_Ratio_Capped']

# Group by cluster and calculate mean of features
cluster_summary = df.groupby('Cluster')[cluster_features].mean()

# Show summary table
st.subheader("Average Features per Cluster")
st.dataframe(cluster_summary.style.format("{:.2f}"))

# Normalize features for plotting
scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(cluster_summary),
    columns=cluster_summary.columns,
    index=cluster_summary.index
).reset_index()

# Convert wide to long format for seaborn plotting
plot_data = normalized.melt(id_vars="Cluster", var_name="Feature", value_name="Value")

# Plot normalized feature comparison
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=plot_data, x='Feature', y='Value', hue='Cluster', ax=ax)
plt.title("Normalized Feature Comparison per Cluster")
plt.xticks(rotation=45)
st.pyplot(fig)

# Plot cluster counts
st.subheader("Number of Customers per Cluster")
cluster_counts = df['Cluster'].value_counts().sort_index()
st.bar_chart(cluster_counts)

