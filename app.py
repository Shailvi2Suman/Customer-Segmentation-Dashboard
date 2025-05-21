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
