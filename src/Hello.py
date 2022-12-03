import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

FILE_PATH_DATA = 'input/heart_2020_cleaned.csv'

def main():
    st.set_page_config(
        page_title="Heart Diseases",
        layout="wide",
        initial_sidebar_state="auto",
        page_icon='heart'
    )
    st.markdown("# Heart Diseases")
    st.markdown("## Introduction")
    st.markdown("""**Center for Disease Control and Prevention (CDC)** states that cardiovascular diseases (CVDs) are one of the leading causes of death in the US as well as globally. Heart diseases lead to about 697,000 deaths in the United States for the year 2020 – which is 1 in every 5th death (1). CVDs cost the country a lot of money and resources and it includes cost such health care services, medicines and loss of productivity due to death. About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking.""", unsafe_allow_html=False)
    st.markdown("""
    How to navigate this web application:
    - To look at the dashboard, select the **Dashboard** page on the left
    - To view/use the predictive model, select the **Predictive Model** page on the left
    """)

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv(FILE_PATH_DATA)
        return data
    

    df = load_data()

if __name__ == '__main__':
    main()