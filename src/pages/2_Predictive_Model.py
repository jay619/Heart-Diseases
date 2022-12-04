import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH_DATA = 'input/heart_2020_cleaned.csv'


def main():
    st.markdown("# Logistic Regression")

    @st.cache(persist=True, allow_output_mutation=True)
    def load_data():
        data = pd.read_csv(FILE_PATH_DATA)
        return data

    df = load_data()
    df.columns = df.columns.str.lower()

    # Encoding target variable
    # df['heartdisease'].replace({'Yes':1,'No':0},inplace=True)

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(df['heartdisease'])


    # Spliting target and feature variables
    target = df['heartdisease']
    features = df.loc[:, df.columns != 'heartdisease']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, train_size=0.8, random_state=7, stratify=df['heartdisease'])
    st.write("##### Train size: ", len(X_train.index), " Test size: ", len(X_test.index))

    ## Model Training

    # Applying MinMax scaler to integer features
    minmax = MinMaxScaler()
    minmax.fit(X_train[['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']], )
    scaled = minmax.transform(X_train[['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']])
    scaled_df = pd.DataFrame(scaled, columns=['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime'])

    # Applying Ordinal Encoding to Yes/No features
    ordinal_enc = OrdinalEncoder(dtype=int)
    ordinal_enc.fit(X_train[['smoking', 'alcoholdrinking', 'stroke', 'diffwalking', 'physicalactivity', 'asthma', 'kidneydisease', 'skincancer']])
    encoded = ordinal_enc.transform(X_train[['smoking', 'alcoholdrinking', 'stroke', 'diffwalking', 'physicalactivity', 'asthma', 'kidneydisease', 'skincancer']])
    ordinal_enc_df = pd.DataFrame(encoded, columns=ordinal_enc.feature_names_in_)

    # One-Hot encoding
    one_hot = OneHotEncoder(categories='auto', drop=None, handle_unknown='ignore', dtype=int)
    one_hot.fit(X_train[['sex', 'agecategory', 'race', 'diabetic', 'genhealth']])
    one_hot_encoded = one_hot.transform(X_train[['sex', 'agecategory', 'race', 'diabetic', 'genhealth']]).toarray()
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot.get_feature_names_out())

    # Concatenated
    X_train_encoded = pd.concat([ordinal_enc_df, scaled_df, one_hot_encoded_df], axis=1)

    ## Model Training & Validation
    clf = LogisticRegression(max_iter=500, class_weight='balanced')
    clf.fit(X_train_encoded, y_train)

    ### Transform Test Data
    test_scaled = minmax.transform(X_test[['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']])
    test_scaled_df = pd.DataFrame(test_scaled, columns=['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime'])
    
    test_encoded = ordinal_enc.transform(X_test[['smoking', 'alcoholdrinking', 'stroke', 'diffwalking', 'physicalactivity', 'asthma', 'kidneydisease', 'skincancer']])
    test_ordinal_enc_df = pd.DataFrame(test_encoded, columns=ordinal_enc.feature_names_in_)

    test_one_hot_encoded = one_hot.transform(X_test[['sex', 'agecategory', 'race', 'diabetic', 'genhealth']]).toarray()
    test_one_hot_encoded_df = pd.DataFrame(test_one_hot_encoded, columns=one_hot.get_feature_names_out())

    X_test_encoded = pd.concat([test_ordinal_enc_df, test_scaled_df, test_one_hot_encoded_df], axis=1)

    y_pred = clf.predict(X_test_encoded)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Model Accuracy: `{:.2%}`\tModel Precision: `{:.2%}`".format(accuracy_score(y_true=y_test, y_pred=y_pred), precision_score(y_true=y_test, y_pred=y_pred, pos_label="Yes")))
        st.markdown("##### Model Recall: `{:.2%}`\tModel F1-Score: `{:.2%}`".format(recall_score(y_true=y_test, y_pred=y_pred, pos_label="Yes"), f1_score(y_true=y_test, y_pred=y_pred, pos_label="Yes")))

    with col2:
        fig, ax = plt.subplots(figsize=(3,3))
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
        sns.heatmap(cm, annot=True, fmt="g", ax=ax)
        st.pyplot(fig=fig)


    # Model Prediction
    st.subheader("Patient Details")

    with st.form(key="patient_details", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            user_bmi = st.number_input(label="Body Mass Index (BMI)", format="%f", min_value=0.0, step=0.1)
            user_smoking = st.selectbox(label="Have you smoked at least 100 cigarettes in your entire life?", options=["-", "Yes", "No"])
            user_alochol = st.selectbox(label="Do you drink alochol?", help="Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week", options=["-", "Yes", "No"])
            user_stroke = st.selectbox(label="Have you ever had a stroke?", options=["-", "Yes", "No"])
            user_physical_health = st.number_input(label="For how many days during the past 30 days was your physical health not good?", help="Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?", format="%d", min_value=0, max_value=30, step=1)
            user_mental_health = st.number_input(label="Thinking about your mental health, for how many days during the past 30 days was your mental health not good?", min_value=0, max_value=30, format="%d", step=1)
            user_sleep = st.number_input(label="On average, how many hours of sleep do you get in a 24-hour period?", format="%d", min_value=0, max_value=24, step=1)
            user_kidney_disease = st.selectbox(label="Any Kidney diseases?", help="Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?", options=["-", "Yes", "No"])
            user_skin_cancer = st.selectbox(label="Do you have or had skin cancer?", options=["-", "Yes", "No"])
            
        with c2:
            user_diff_walking = st.selectbox(label="Do you have serious difficulty walking or climbing stairs?", options=["-", "Yes", "No"])
            user_sex = st.selectbox(label="Sex", options=["-", "Male", "Female"])
            user_age_cat = st.selectbox(label="Age Category", options=["-", "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
            user_race = st.selectbox(label="Race", options=["-","American Indian/Alaskan Native","Asian","Black","Hispanic","Other","White"])
            user_diabetic = st.selectbox(label="Do you have or had diabetes?", options=["-", "Yes", "No"])
            user_phy_activity = st.selectbox(label="Have you had any physical activity in the past 30 days?", options=["-", "Yes", "No"], help="Adults who reported doing physical activity or exercise during the past 30 days other than their regular job")
            user_gen_health = st.selectbox(label="How is your general health?", options=["-", "Excellent", "Fair", "Good", "Poor", "Very good"])
            user_asthama = st.selectbox(label="Do you have or had asthama?", options=["-", "Yes", "No"])

        is_predict = st.form_submit_button(label="Predict", type="primary")
        # Make the prediction based on user inputs
        if is_predict:
            user_inp_scaled = minmax.transform([[user_bmi, user_physical_health, user_mental_health, user_sleep]])
            user_inp_ordinal = ordinal_enc.transform([[user_smoking, user_alochol, user_stroke, user_diff_walking, user_phy_activity, user_asthama, user_kidney_disease, user_skin_cancer]])
            user_inp_one_hot = one_hot.transform([[user_sex, user_age_cat, user_race, user_diabetic, user_gen_health]]).toarray()

            user_input = np.concatenate([user_inp_scaled, user_inp_ordinal, user_inp_one_hot], axis=1)
            user_inp_pred = clf.predict(user_input)[0]
    
    
    if user_inp_pred == "Yes":
        st.markdown("### Does user have a high chance of heart diseases? Based on the inputs, we think: `{}` 🚨".format(user_inp_pred))
    else:
        st.markdown("### Does user have a high chance of heart diseases? Based on the inputs, we think: `{}` ✅".format(user_inp_pred))
            # label_encoder.inverse_transform(user_inp_pred)[0]

    

if __name__ == '__main__':
    main()