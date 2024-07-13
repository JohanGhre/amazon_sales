import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

# Streamlit setup
st.set_page_config(layout="wide")

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load data
sales_data = load_data("AmazonSalesData.csv")

# Sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.multiselect("Select Region", options=sales_data["Region"].unique(), default=sales_data["Region"].unique())
selected_country = st.sidebar.multiselect("Select Country", options=sales_data["Country"].unique(), default=sales_data["Country"].unique())
selected_item = st.sidebar.multiselect("Select Item Type", options=sales_data["Item Type"].unique(), default=sales_data["Item Type"].unique())
selected_channel = st.sidebar.multiselect("Select Sales Channel", options=sales_data["Sales Channel"].unique(), default=sales_data["Sales Channel"].unique())
selected_priority = st.sidebar.multiselect("Select Order Priority", options=sales_data["Order Priority"].unique(), default=sales_data["Order Priority"].unique())

# Convert 'Order Date' to 'YearMonth'
sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'])
sales_data['YearMonth'] = sales_data['Order Date'].dt.to_period('M').astype(str)  # Convert to string

# Filter data based on selections
filtered_data = sales_data[
    (sales_data["Region"].isin(selected_region)) &
    (sales_data["Country"].isin(selected_country)) &
    (sales_data["Item Type"].isin(selected_item)) &
    (sales_data["Sales Channel"].isin(selected_channel)) &
    (sales_data["Order Priority"].isin(selected_priority))
]

# Renommer les colonnes pour enlever tout caractère indésirable
filtered_data.columns = [col.strip() for col in filtered_data.columns]

# Function to create bar chart
def create_bar_chart(data, x, y, title, color):
    fig = px.bar(data, x=x, y=y, title=title)
    fig.update_traces(marker_color=color, marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False)  # Remove vertical gridlines and zero line
    fig.update_yaxes(showgrid=False, zeroline=False)  # Remove horizontal gridlines and zero line
    return fig

# Function to create line chart
def create_line_chart(data, x, y, title, colors=None):
    fig = go.Figure()
    if colors is None:
        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)']

    for i, col in enumerate(y):
        fig.add_trace(go.Scatter(x=data[x], y=data[col], mode='lines', name=col, line=dict(color=colors[i], width=2)))
    
    fig.update_layout(title=title, showlegend=True, hovermode='closest', width=600, height=400)
    fig.update_xaxes(showgrid=False, zeroline=False)  # Remove vertical gridlines and zero line
    fig.update_yaxes(showgrid=False, zeroline=False)  # Remove horizontal gridlines and zero line
    return fig

# Function to load the Ridge model
def load_model():
    with open('ridge_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict with model
def predict_with_model(model, X):
    return model.predict(X)

# Function to create regression plot
def create_regression_plot(X_test, y_test, y_pred):
    coefficients = np.polyfit(X_test['Total Revenue'], y_test, 1)
    line = np.polyval(coefficients, X_test['Total Revenue'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test['Total Revenue'], y=y_test, mode='markers', name='Actual Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=X_test['Total Revenue'], y=y_pred, mode='markers', name='Predictions', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=X_test['Total Revenue'], y=line, mode='lines', name='Linear Regression', line=dict(color='green')))
    fig.update_layout(title='Profit Prediction based on Revenue (Ridge Regression)',
                      xaxis_title='Total Revenue', yaxis_title='Total Profit', showlegend=True, hovermode='closest',
                      width=600, height=400)
    fig.update_xaxes(showgrid=False, zeroline=False)  # Remove vertical gridlines and zero line
    fig.update_yaxes(showgrid=False, zeroline=False)  # Remove horizontal gridlines and zero line
    
    return fig

# Load Ridge model
ridge_model = load_model()

# Left column: Sales Overview
st.title("Sales Overview")

col1, col2 = st.columns(2)

with col1:
    st.header("Regional Sales")
    fig_region = create_bar_chart(filtered_data.groupby('Region').agg({'Total Revenue': 'sum'}).reset_index(),
                                  x='Region', y='Total Revenue', title='Total Revenue by Region', color='rgba(0, 0, 255, 0.5)')
    st.plotly_chart(fig_region, use_container_width=True)

    st.header("Sales by Item Type")
    fig_item = create_bar_chart(filtered_data.groupby('Item Type').agg({'Total Revenue': 'sum'}).reset_index(),
                                x='Item Type', y='Total Revenue', title='Total Revenue by Item Type', color='rgba(255, 0, 0, 0.5)')
    st.plotly_chart(fig_item, use_container_width=True)

with col2:
    st.header("Sales Channel Analysis")
    fig_sales_channel = create_bar_chart(filtered_data.groupby('Sales Channel').agg({'Total Revenue': 'sum'}).reset_index(),
                                         x='Sales Channel', y='Total Revenue', title='Total Revenue by Sales Channel', color='rgba(0, 128, 0, 0.5)')
    st.plotly_chart(fig_sales_channel, use_container_width=True)

    st.header("Profit Prediction")
    X = filtered_data[['Unit Price', 'Units Sold', 'Total Revenue']]
    y = filtered_data['Total Profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_pred = predict_with_model(ridge_model, X_test)
    fig_regression = create_regression_plot(X_test, y_test, y_pred)
    st.plotly_chart(fig_regression, use_container_width=True)

# Right column: Financial Insights
st.title("Financial Insights")

st.header("Temporal Analysis")
if 'YearMonth' in filtered_data.columns and 'Total Revenue' in filtered_data.columns:
    fig_temporal = create_line_chart(filtered_data.groupby('YearMonth').agg({'Total Revenue': 'sum'}).reset_index(),
                                     x='YearMonth', y=['Total Revenue'], title='Revenue Trends over Time', colors=['rgb(255, 165, 0)'])
    st.plotly_chart(fig_temporal, use_container_width=True)
else:
    st.warning("Data does not contain 'YearMonth' or 'Total Revenue' column. Check your data preprocessing steps.")

st.header("Profit Margin Evolution")
if 'YearMonth' in filtered_data.columns and 'Total Revenue' in filtered_data.columns and 'Total Profit' in filtered_data.columns:
    filtered_data['Profit Margin'] = filtered_data['Total Profit'] / filtered_data['Total Revenue']
    fig_margin = create_line_chart(filtered_data.groupby('YearMonth').agg({'Profit Margin': 'mean'}).reset_index(),
                                   x='YearMonth', y=['Profit Margin'], title='Profit Margin Evolution', colors=['rgb(75, 0, 130)'])
    st.plotly_chart(fig_margin, use_container_width=True)
else:
    st.warning("Data does not contain necessary columns for Profit Margin Evolution.")

st.header("Revenue and Cost Evolution")
if 'YearMonth' in filtered_data.columns and 'Total Revenue' in filtered_data.columns and 'Total Cost' in filtered_data.columns:
    fig_costs_revenue = create_line_chart(filtered_data.groupby('YearMonth').agg({'Total Revenue': 'sum', 'Total Cost': 'sum'}).reset_index(),
                                          x='YearMonth', y=['Total Revenue', 'Total Cost'], title='Revenue and Cost Evolution',
                                          colors=['rgb(255, 0, 0)', 'rgb(0, 128, 0)'])
    st.plotly_chart(fig_costs_revenue, use_container_width=True)
else:
    st.warning("Data does not contain necessary columns for Revenue and Cost Evolution.")


# Calcul des statistiques
total_revenue = filtered_data['Total Revenue'].sum()
total_units_sold = filtered_data['Units Sold'].sum()
average_profit = filtered_data['Total Profit'].mean()
total_cost = filtered_data['Total Cost'].sum()
profit_margin = (filtered_data['Total Profit'].sum() / filtered_data['Total Revenue'].sum()) * 100
num_orders = filtered_data['Order ID'].nunique()
avg_unit_price = filtered_data['Unit Price'].mean()

# Affichage des statistiques dans une bande
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Revenue", f"{total_revenue:.0f}")

with col2:
    st.metric("Total Units Sold", f"{total_units_sold:.0f}")

with col3:
    st.metric("Average Profit", f"${average_profit:.2f}")

with col4:
    st.metric("Total Cost", f"${total_cost:.0f}")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Profit Margin (%)", f"{profit_margin:.2f}%")

with col6:
    st.metric("Number of Orders", f"{num_orders:,}")

with col7:
    st.metric("Average Unit Price", f"${avg_unit_price:.2f}")





# Add prediction section
st.header("Profit Prediction")
st.write("Enter the details to predict profit:")

unit_price = st.number_input("Unit Price", min_value=0)
units_sold = st.number_input("Units Sold", min_value=0)
total_revenue = st.number_input("Total Revenue", min_value=0)

prediction_input = np.array([[unit_price, units_sold, total_revenue]])
if st.button("Predict"):
    predicted_profit = predict_with_model(ridge_model, prediction_input)[0]
    st.write(f"Predicted Total Profit: {predicted_profit:.2f}")
