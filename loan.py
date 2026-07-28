import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sb
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
import streamlit as st
import plotly.express as px
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="LOAN PREDICTION APP", layout='wide', initial_sidebar_state='expanded')

st.title("LOAN PREDICTION APP")

st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")

from PIL import Image  # Import Image from Pillow
img = Image.open("selected-1-1.webp")  # Open the image file
st.image(img, width=1200)

st.sidebar.title("Select Visual Charts")
st.sidebar.markdown("Select the Charts/Plots accordingly:")
st.markdown(
    """
    <style>
    .stApp {
        color:black;
        text-align:center;
        background-color:skyblue;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #4A90E2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("LOADING DATASET")

df = pd.read_csv('loan_data (1).csv')
st.write(df.head(50))

chart_visual = st.sidebar.selectbox('Select Charts/Plot type',
                                     ('Line Chart', 'Bar Chart', 'Bubble Chart'))
st.sidebar.checkbox("Show Analysis", True, key=1)
selected_status = st.sidebar.selectbox('Select Status', options=['Loan_ID', 'Gender', 'Married', 'Loan_Status'])

selected_status1 = st.sidebar.selectbox('Loading', options=['Shape', 'Information', 'Get values',
                                                              'Expolatory Data Values', 'Barchart',
                                                              'Observation', 'Outliers', 'Mean Amount'])

fig = go.Figure()

if chart_visual == 'Line Chart':
    if selected_status == 'Loan_ID':
        fig.add_trace(go.Scatter(x=df.ApplicantIncome, y=df.Loan_ID, mode='lines', name='Loan_ID'))
    if selected_status == 'Gender':
        fig.add_trace(go.Scatter(x=df.ApplicantIncome, y=df.Gender, mode='lines', name='Gender'))
    if selected_status == 'Married':
        fig.add_trace(go.Scatter(x=df.ApplicantIncome, y=df.Married, mode='lines', name='Married'))
    if selected_status == 'Loan_Status':
        fig.add_trace(go.Scatter(x=df.ApplicantIncome, y=df.Loan_Status, mode='lines', name="Loan_Status"))

elif chart_visual == 'Bar Chart':
    if selected_status == 'Loan_ID':
        fig.add_trace(go.Bar(x=df.ApplicantIncome, y=df.Loan_ID, name='Loan_ID'))
    if selected_status == 'Gender':
        fig.add_trace(go.Bar(x=df.ApplicantIncome, y=df.Gender, name='Gender'))
    if selected_status == 'Married':
        fig.add_trace(go.Bar(x=df.ApplicantIncome, y=df.Married, name='Married'))
    if selected_status == 'Loan_Status':
        fig.add_trace(go.Bar(x=df.ApplicantIncome, y=df.Loan_Status, name="Loan_Status"))

elif chart_visual == 'Bubble Chart':
    if selected_status == 'Loan_ID':
        fig.add_trace(go.Scatter(x=df.ApplicantIncome, y=df.Loan_ID, mode='markers',
                                  marker_size=[40, 60, 80, 60, 40, 50], name='Loan_ID'))
    if selected_status == 'Gender':
        fig.add_trace(go.Scatter(x=df.ApplicantIncome, y=df.Gender, mode='markers',
                                  marker_size=[40, 60, 40, 60, 40, 60], name='Gender'))
    if selected_status == 'Married':
        fig.add_trace(go.Scatter(x=df.ApplicantIncome, y=df.Married, mode='markers',
                                  marker_size=[40, 60, 80, 60, 40, 50], name='Married'))
    if selected_status == 'Loan_Status':
        fig.add_trace(go.Scatter(x=df.ApplicantIncome, y=df.Loan_Status, mode='markers',
                                  marker_size=[40, 60, 80, 60, 40, 50], name="Loan_Status"))

st.plotly_chart(fig, use_container_width=True)

st.subheader("SHAPE OF THE DATASET")
if selected_status1 == "Shape":
    st.write(df.shape)

st.subheader("INFORMATION OF THE DATASET")
if selected_status1 == "Information":
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

st.subheader("GET VALUES")
if selected_status1 == "Get values":
    st.write(df.describe())

st.subheader("EXPOLATORY DATA ANALYSIS")
if selected_status1 == "Expolatory Data Values":
    temp = df['Loan_Status'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(temp.values, labels=temp.index, autopct='%1.1f%%')
    st.pyplot(fig1)

st.subheader("BAR CHART")
if selected_status1 == "Barchart":
    fig2, ax2 = plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['Gender', 'Married']):
        plt.subplot(1, 2, i + 1)
        sb.countplot(data=df, x=col, hue='Loan_Status')
    st.pyplot(fig2)

st.subheader("OBSERVATIONS")
if selected_status1 == "Observation":
    fig3, ax3 = plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
        plt.subplot(1, 2, i + 1)
        sb.histplot(df[col].dropna(), kde=True)
    st.pyplot(fig3)

st.subheader("OUTLIERS")
if selected_status1 == "Outliers":
    fig4, ax4 = plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
        plt.subplot(1, 2, i + 1)
        sb.boxplot(df[col])
    st.pyplot(fig4)

st.subheader("MEAN AMOUNT OF THE LOAN GRANTED")

if selected_status1 == "Mean Amount":
    df_filtered = df[df['ApplicantIncome'] < 25000]
    df_filtered = df_filtered[df_filtered['LoanAmount'] < 400000]
    st.write(df_filtered.groupby('Gender').mean(numeric_only=True)['LoanAmount'])
    st.write(df_filtered.groupby(['Married', 'Gender']).mean(numeric_only=True)['LoanAmount'])

# =========================================================
# like Gender, Married, Credit_History, LoanAmount break both
# LabelEncoder and SVC. Fill them BEFORE encoding.
# =========================================================
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].astype(str)
        df[col] = df[col].fillna(df[col].mode()[0])


def encode_labels(data):
    data = data.copy()
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data


# Keep a human-readable copy (before label encoding) so we can later show
# predictions with the original Gender/Married/Loan_ID text instead of numbers.
df_display = df.copy()

st.subheader("Heat Map")
fig7, ax7 = plt.subplots(figsize=(15, 5))
df = encode_labels(df)
sb.heatmap(df.drop(columns=['Loan_ID'], errors='ignore').corr(numeric_only=True),
           ax=ax7, annot=True, cmap='coolwarm')
st.pyplot(fig7)

features = df.drop(['Loan_ID', 'Loan_Status'], axis=1, errors='ignore')
target = df['Loan_Status'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                    test_size=0.2,
                                                    random_state=10)
val_index = X_val.index  # remember original row indices before scaling turns X_val into a numpy array

# As the data was highly imbalanced we will balance
# it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)
st.write(X_train.shape, X.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

model = SVC(kernel='rbf')
model.fit(X, Y)
st.write('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X)))
st.write('Validation Accuracy : ', metrics.roc_auc_score(Y_val, model.predict(X_val)))

st.subheader("MODEL EVALUATION")
training_roc_auc = roc_auc_score(Y, model.predict(X))
validation_roc_auc = roc_auc_score(Y_val, model.predict(X_val))
st.write('Training ROC AUC Score:', training_roc_auc)
st.write('Validation ROC AUC Score:', validation_roc_auc)

st.subheader("Confusion Matrix")
cm = confusion_matrix(Y_val, model.predict(X_val))
fig8, ax8 = plt.subplots(figsize=(6, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
ax8.set_title('Confusion Matrix')
ax8.set_xlabel('Predicted Label')
ax8.set_ylabel('True Label')
st.pyplot(fig8)

st.subheader("Classification Report")
# validation set (data leakage / meaningless report). Now we just use
# the already-trained model's predictions on X_val.
report = classification_report(Y_val, model.predict(X_val))
st.code(report)


# shows how many are predicted Approved/Rejected, plus total loan amount.
# =========================================================
st.subheader("LOAN APPROVAL PREDICTIONS")

val_predictions = model.predict(X_val)
result_df = df_display.loc[val_index].copy()
result_df['Predicted_Loan_Status'] = np.where(val_predictions == 1, 'Approved', 'Rejected')

approved_count = int((result_df['Predicted_Loan_Status'] == 'Approved').sum())
rejected_count = int((result_df['Predicted_Loan_Status'] == 'Rejected').sum())
total_loan_approved = result_df.loc[result_df['Predicted_Loan_Status'] == 'Approved', 'LoanAmount'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Applicants", len(result_df))
col2.metric("Predicted Approved", approved_count)
col3.metric("Predicted Rejected", rejected_count)

st.write(f"Total loan amount likely to be granted (approved applicants): **{total_loan_approved:.2f}**")

st.write("Detailed predictions:")
st.dataframe(result_df[['Loan_ID', 'Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Predicted_Loan_Status']])

csv_download = result_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Predictions as CSV", data=csv_download,
                    file_name="loan_predictions.csv", mime="text/csv")

st.title("END OF THE PROJECT")