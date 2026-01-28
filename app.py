

import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title='Credit Card Default Prediction', page_icon='💳', layout='centered')

with st.sidebar:
    st.image('https://cdn-icons-png.flaticon.com/512/633/633611.png', width=80)
    st.title('Credit Default Predictor')
    st.markdown('---')
    st.write('Fill in the customer details below:')

st.header('Credit Card Default Prediction')
st.write('Enter customer information to predict the probability of defaulting next month.')

# Load the trained model
model = joblib.load('credit_default_model.pkl')

with st.form('prediction_form'):
    st.subheader('Personal & Account Information')
    col1, col2 = st.columns(2)
    with col1:
        LIMIT_BAL = st.number_input('Credit Limit (NT dollar)', min_value=0, value=20000, help='Total credit limit for the account')
        SEX = st.selectbox('Gender', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female', help='1=Male, 2=Female')
        EDUCATION = st.selectbox('Education', [1, 2, 3, 4], format_func=lambda x: ['Graduate School', 'University', 'High School', 'Other'][x-1], help='1=Grad, 2=Univ, 3=HS, 4=Other')
        MARRIAGE = st.selectbox('Marital Status', [1, 2, 3], format_func=lambda x: ['Married', 'Single', 'Other'][x-1], help='1=Married, 2=Single, 3=Other')
        AGE = st.number_input('Age', min_value=18, max_value=100, value=30, help='Age in years')
    with col2:
        PAY_0 = st.number_input('Repayment Status (Sept)', min_value=-2, max_value=8, value=0, help='-2=No consumption, 0=Paid in full, 1=1 month late, etc.')
        PAY_2 = st.number_input('Repayment Status (Aug)', min_value=-2, max_value=8, value=0, help='Repayment status in August')
        PAY_3 = st.number_input('Repayment Status (July)', min_value=-2, max_value=8, value=0, help='Repayment status in July')
        PAY_4 = st.number_input('Repayment Status (June)', min_value=-2, max_value=8, value=0, help='Repayment status in June')
        PAY_5 = st.number_input('Repayment Status (May)', min_value=-2, max_value=8, value=0, help='Repayment status in May')
        PAY_6 = st.number_input('Repayment Status (April)', min_value=-2, max_value=8, value=0, help='Repayment status in April')

    st.subheader('Bill Statement Amounts')
    bill_cols = st.columns(3)
    BILL_AMT1 = bill_cols[0].number_input('Bill Amt (Sept)', value=0, help='Bill statement in September')
    BILL_AMT2 = bill_cols[1].number_input('Bill Amt (Aug)', value=0, help='Bill statement in August')
    BILL_AMT3 = bill_cols[2].number_input('Bill Amt (July)', value=0, help='Bill statement in July')
    BILL_AMT4 = bill_cols[0].number_input('Bill Amt (June)', value=0, help='Bill statement in June')
    BILL_AMT5 = bill_cols[1].number_input('Bill Amt (May)', value=0, help='Bill statement in May')
    BILL_AMT6 = bill_cols[2].number_input('Bill Amt (April)', value=0, help='Bill statement in April')

    st.subheader('Amount Paid')
    pay_cols = st.columns(3)
    PAY_AMT1 = pay_cols[0].number_input('Paid (Sept)', value=0, help='Amount paid in September')
    PAY_AMT2 = pay_cols[1].number_input('Paid (Aug)', value=0, help='Amount paid in August')
    PAY_AMT3 = pay_cols[2].number_input('Paid (July)', value=0, help='Amount paid in July')
    PAY_AMT4 = pay_cols[0].number_input('Paid (June)', value=0, help='Amount paid in June')
    PAY_AMT5 = pay_cols[1].number_input('Paid (May)', value=0, help='Amount paid in May')
    PAY_AMT6 = pay_cols[2].number_input('Paid (April)', value=0, help='Amount paid in April')

    submitted = st.form_submit_button('Predict Default Probability')

if 'submitted' in locals() and submitted:
    input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
        BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
        PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    st.markdown('---')
    if prediction[0] == 1:
        st.error(f'⚠️ Prediction: Will Default')
        st.progress(int(probability*100))
        st.write(f'**Probability of Default:** {probability:.2%}')
    else:
        st.success(f'✅ Prediction: Will Not Default')
        st.progress(int((1-probability)*100))
        st.write(f'**Probability of Default:** {probability:.2%}')
