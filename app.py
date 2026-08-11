import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection - Demo")
st.write("Note: Idhi demo dataset. Real project lo 284k rows unna creditcard.csv use chestham")

@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 5000
    n_fraud = 25 
    data = {}
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    data['Amount'] = np.random.exponential(100, n_samples)
    data['Time'] = np.arange(n_samples)
    data['Class'] = [0]*(n_samples-n_fraud) + [1]*n_fraud
    np.random.shuffle(data['Class'])
    return pd.DataFrame(data)

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Transactions", len(df))
with col2:
    st.metric("Fraud Cases", df['Class'].sum())

st.write("### Fraud vs Normal Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Class', data=df, ax=ax)
ax.set_xticklabels(['Normal', 'Fraud'])
st.pyplot(fig)

st.write("### Model Training - RandomForest")
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

st.write("**Accuracy:**", round(accuracy_score(y_test, preds), 4))
st.text(classification_report(y_test, preds))

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, preds)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
st.pyplot(fig2)
