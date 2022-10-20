# Import libraries
import streamlit as st

# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plost
from itertools import combinations

# sklearn tools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 


# Page configuration
st.set_page_config(page_title=None, page_icon="chart_with_upwards_trend", layout="centered")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load dataset
data = pd.read_csv("heart.csv")
data.sample(frac=1) # shuffle data

# set target and disease name
disease_name = "Heart Failure"
target = "HeartDisease"
# categorical columns
categorical_columns = list(data.select_dtypes("object").columns) + ["HeartDisease"]

# numerical columns
numeric_columns = list(data.select_dtypes(exclude = ['object']).columns)
numeric_columns.remove("HeartDisease")

group_labels = list(data[target].unique())
label_map = {"Heart Failure" : 1, "No Heart Failure":0}
label_map_r = {1:"Heart Failure", 0: "No Heart Failure"}

# convert all numeric columns to floats
for column in numeric_columns:
    data[column] = data[column].astype(float)
    
# units of numerix columns
units = {"Age" : "years", "RestingBP" : "mmHg", "Cholesterol": "mm/dL",
         "FastingBS" : "mg/dL", "MaxHR" : "BPM", "Oldpeak" : "ST"}

# begin building sidebar
st.sidebar.title("Navigation")
# setting options for showing data summary for each class in the dataset or all the dataset
option = st.sidebar.radio("Data Summary",["All"]+group_labels)
if option == "All":
    usage_data = data
else:
    usage_data = data[data["HeartDisease"] == label_map[option]]

# dashboard for numeric data
st.caption("**Dataset Summary**")
st_numeric_columns = []
N_numeric = len(numeric_columns)
while N_numeric > 0:
    c1, c2, c3 = st.columns(3)
    st_numeric_columns += [c1,c2,c3]
    N_numeric -= 3
    
# create space
st.text(" ");st.text(" ");

# dashboard for categorcial data
st_categorical_columns = []
N_categorical = len(categorical_columns)
while N_categorical > 0:
    c1, c2, c3, c4 = st.columns(4)
    st_categorical_columns += [c1,c2,c3,c4]
    N_categorical -= 4

# plot numeric columns
table_columns = usage_data.columns[:-1]
for i, column in enumerate(numeric_columns):
    mean = round(usage_data[column].mean(),2)
    std = round(usage_data[column].std(),2)
    unit = units[column]
    st_numeric_columns[i].metric(column, f"{mean} {unit}", f"± {std} {unit}")
    
# plot categorical columns
for i, column in enumerate(categorical_columns):
    # control data summary display
    if column == target:
        usable_data = data.copy()
    else:
        usable_data = usage_data.copy()
        
    with st_categorical_columns[i]:
        categorical_labels = list(usable_data[column].unique())
        sub_data = {column: categorical_labels, "Count" : [0]*len(usable_data[column].unique())}
        for item in usable_data[column]:
            sub_data["Count"][categorical_labels.index(item)] += 1
        sub_data_df = pd.DataFrame(sub_data)
        plost.donut_chart(
            data=sub_data_df,
            theta='Count',
        color=column)

# Visualize Categorical Features
categorical_colors = ["salmon", "lightblue", "orange", "navy"]
st.caption("**Categorical - Bar charts**")
selected_categorical_column = st.sidebar.radio("Categorical - Bar charts",categorical_columns)
# Create a plot of crosstab
feature_classes = data[selected_categorical_column].unique()
ax = pd.crosstab(data[target], 
                 data[selected_categorical_column ]).plot(kind="bar",
                                                          figsize=(10, 6),
                                                          color=categorical_colors[:len(feature_classes)])
plt.title(f"Heart Failure Frequency for {selected_categorical_column}")
plt.xlabel("0 = No Diesease, 1 = Disease")
plt.ylabel("Amount")
plt.legend();
plt.xticks(rotation=0);
figure = ax.get_figure()
st.pyplot(figure)


# Visualize Numeric Features
st.caption("**Numeric - Bar charts**")
selected_categorical_column = st.sidebar.radio("Numeric - Histogram Plots",numeric_columns)

figure,ax = plt.subplots(figsize=(10,6))
data_disease = data[data['HeartDisease']==1]
data_no_disease = data[data['HeartDisease']==0]

sns.distplot(data_no_disease[selected_categorical_column], label='No disease')
sns.distplot(data_disease[selected_categorical_column], label='Disease')
ax.legend()
ax.set_title(f'{selected_categorical_column} Distribution');
st.pyplot(figure)

# Pairplots Between Features
st.caption("**Pairwise - Scatter plots**")
st.sidebar.caption("**Piarwise- Scatter plots**")
# encode categorical variables
catCols = data.select_dtypes("object").columns
for column in catCols: data[column] = data[column].astype("category").cat.codes
col1, col2 = st.sidebar.columns(2)
x = col1.selectbox("x-axis", options = data.columns[:-1], index = 0, key = 1)
y = col2.selectbox("y-axis", options = data.columns[:-1], index = 1, key = 2)
# Create a figure
figure = plt.figure(figsize=(10, 6))
ax1 =plt.scatter(data[x][data[target]==1],
            data[y][data[target]==1],
            c="salmon")
# Scatter with negative examples
ax2 =plt.scatter(data[x][data[target]==0],
            data[y][data[target]==0],
            c="lightblue")
# Add some helpful info
plt.title(f"{disease_name} in function of {x} and {y}")
plt.xlabel(x)
plt.ylabel(y)
plt.legend([ax1,ax2],group_labels);
st.pyplot(figure)

# PCA
st.sidebar.caption("**Do you want to see PCA results?**")           
button = st.sidebar.button("PCA", key = 3)


if button:
    # scale data
    scaled = StandardScaler() 
    scaled.fit(data) 
    scaled_data = scaled.transform(data) 
    
    # 2 component PCA
    st.caption("**2D PCA**")
    pca = PCA(n_components=2) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    # 
    labels = data.HeartDisease
    # plot
    PC1 = 0 
    PC2 = 1 
    figure = plt.figure(figsize=(10, 6))
    for i in range(0, xpca.shape[0]): 
        if labels[i] == 0:
            c1 = plt.scatter(xpca[i,PC1],xpca[i,PC2], c='lightblue', marker='o')
        elif labels[i] == 1:
            c2 = plt.scatter(xpca[i,PC1],xpca[i,PC2], c='salmon', marker='+') 

    # Add some helpful info
    plt.title(f"PCA Analysis of {disease_name} Dataset")
    plt.xlabel("PCA_1")
    plt.ylabel("PCA_2")
    plt.legend([c1,c2],["No Disease", "Disease"]);
    st.pyplot(figure) 
                   
    # 3 component PCA
    st.caption("**3D PCA**")
    pca = PCA(n_components=3) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)  
    pca_data = pd.concat([pd.DataFrame(xpca), data.HeartDisease], axis = 1)
    figure = px.scatter_3d(pca_data, x=0, y=1, z=2, color = "HeartDisease",
                   labels={"0": 'PC 1', "1": 'PC 2', "2": 'PC 3'})
    st.plotly_chart(figure)