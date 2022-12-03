import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import seaborn as sns
import streamlit as st

FILE_PATH_DATA = 'input/heart_2020_cleaned.csv'

def main():
    st.set_page_config(
        page_title="Heart Diseases",
        layout="wide",
        initial_sidebar_state="auto"
    )
    st.title("Heart Diseases")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv(FILE_PATH_DATA)
        return data

    df = load_data()

if __name__ == '__main__':
    main()