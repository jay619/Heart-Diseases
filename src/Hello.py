import streamlit as st
import streamlit.components.v1 as components


def main():
    st.set_page_config(
        page_title="Heart Diseases Key Factors",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon='heart'
    )
    st.markdown("# Heart Diseases")
    st.markdown("## Introduction")
    st.markdown("""**Center for Disease Control and Prevention (CDC)** states that cardiovascular diseases (CVDs) are one of the leading causes of death in the US as well as globally. Heart diseases lead to about 697,000 deaths in the United States for the year 2020 – which is 1 in every 5th death (1). CVDs cost the country a lot of money and resources and it includes cost such health care services, medicines and loss of productivity due to death. About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking.""", unsafe_allow_html=False)
    st.markdown("""
    How to navigate this web application:
    - Expan the Sidebar by clicking on the arrow (▶️) on the top left of the screen
    - To look at the dashboard, select the **Dashboard** page on the left
    - To view the predictive models performace or use the model to make predictions, select the **Predictive Model** page on the left
    Using the data from the surveys conducted, one can learn about an individual’s health history and use that to build an early detection system to prevent an individual from having a cardiovascular disease such as heart failure.
    """)


    st.markdown('For any questions or bugs, please <a href="mailto:jshah40@jh.edu">Send email</a>', unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()