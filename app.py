import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

@st.cache_data
def load_data():
    # Kaggle nunchi direct download
    url = "https://storage.googleapis.com/kaggle-datasets-185220/creditcardfraud/creditcard.csv"
    df = pd.read_csv(url)
    return df

with st.spinner("Loading dataset from Kaggle... 1 min padthundi"):
    df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())

st.write("### Fraud vs Normal Transactions")
fig, ax = plt.subplots()
sns.countplot(x='Class', data=df, ax=ax)
ax.set_xticklabels(['Normal', 'Fraud'])
st.pyplot(fig)

st.write("### Model Training")
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

st.write("**Accuracy:**", accuracy_score(y_test, preds))
st.text(classification_report(y_test, preds))