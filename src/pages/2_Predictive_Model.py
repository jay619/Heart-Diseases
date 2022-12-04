import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

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
    df['heartdisease'].replace({'Yes':1,'No':0},inplace=True)

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

    # clf = LogisticRegression(max_iter=500, class_weight='balanced')
    # clf.fit(X_train_encoded, y_train)

    ## Model Training Validation


    # Model Prediction

    st.sidebar.subheader("Patient Details")
    # is_predict = st.sidebar.button(label="Predict")

    with st.sidebar.form(key="patient_details", clear_on_submit=False):
        user_bmi = st.number_input(label="Body Mass Index (BMI)", format="%f", min_value=0.0, step=0.1)

        st.form_submit_button(label="Predict", type="primary")

    

if __name__ == '__main__':
    main()