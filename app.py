import streamlit as st
from multiapp import MultiApp
from apps import bank_marketing_streamlit,data_analysis  # import your app modules here

app = MultiApp()
# team1 = st.secrets["abc"]
# Add all your application here
app.add_app("Data Analysis", data_analysis.main)
app.add_app("Prediction", bank_marketing_streamlit.main)
# The main app
app.run()
#streamlit run app.py
