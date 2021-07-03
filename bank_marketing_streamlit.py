import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pickle
import sys
from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
        
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import base64

main_bg = "D://photo.jfif"
main_bg_ext = "jfif"



st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    """,
    unsafe_allow_html=True
)

print(sys.path)
data=pd.read_csv("D:\\bank-additional-full.csv", sep = ";")
def main():
    
    st.title("Bank Marketing Prediction")
    
    htk=  """
    <div style="background-color:#004d99;padding:0px">
    <h2 style="color:white;text-align:center;">CUSTOMER DESPOIT PREDICTION APP </h2>
    </div>
    """
    st.markdown(htk,unsafe_allow_html=True)
    
    if st.button("Problem Statement"):
       
        st.markdown(""" The  goal is to make a predictive model to predict if the customer will respond positively to the
            campaign organised by a portugese bank institution. Often, more than one contact to the same client was required, in order to access 
        if the product (bank term deposit) would be subscribed or not subscribed.
        This is a tedious task to do and will consume much time ,the model can help in avoiding both.
         """)
    if st.checkbox("Data Description"):
        st.text(
        """ 
        The data is related with direct marketing campaigns of a Portuguese banking institution.
        The marketing campaigns were based on phone calls.All the information can be found in the link given below.
        Please refer to the data set link before actually going forward with the app to get a better understanding and clear idea.
        """)
          
        link='[Data set link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)'
        st.markdown(link,unsafe_allow_html=True)


    st.sidebar.header("GRADIENT BOOST CLASSIFIER")   
    age=st.slider("Enter age of the customer",18,95)
    age=int(age)
    # age=int(scaler.fit_transform([[age]]))

    job=st.selectbox("Enter the type of job customer do",("select",'housemaid', 'services', 'admin.', 'blue-collar', 'technician',
    'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'))
    job=int(encoder.fit_transform([[job]]))
        
    marital=st.selectbox("what is customer's marital status?",("select",'married','single' ,'divorced'))
    
    education=st.selectbox("Enter customer's education level",("select",'basic.4y', 'basic.6y' ,'basic.9y' ,'high.school','professional.course','university.degree','unknown','illiterate'))
    
    #targeted=st.selectbox("Do the customer have target?", ("select",'yes' , 'no'))
    
    default=st.selectbox("Do the customer have credit in default?", ("select",'yes' , 'no'))
            
    housing=st.selectbox("Do the customer have Housing Loan?", ("select",'yes' , 'no'))
    
    loan=st.selectbox("Do the customer have Personal Loan?",("select",'yes', 'no'))
    
    contact=st.radio("How do you prefer to communicate",("select","unknown",'telephone', 'cellular'))

    month=st.selectbox("Which month the customer was last contacted in?",("select",'jan',"feb","mar",'apr',"may","jun","jul","aug","sep","oct","nov","dec"))
    
    day_of_week=st.radio("Which day of week the customer was contacted in",("select","mon",'tue', 'wed','thu','fri'))

    duration=st.text_input("Enter last contact duration with the customer in sec?",0,4918)
    if not duration:
        st.warning("Enter Duration Period")
        
    else:
        if duration.isalpha() and duration.isalnum():
            st.warning("Please enter an integer number")
            pass
        else:
            duration=int(duration)
            if duration>4918 or duration<0:
                st.warning("Please enter an number between 0 & 4918")
            # duration=int(scaler.fit_transform([[duration]]))


    campaign=st.slider("Enter number of contacts performed during this campaign and for this client",1,63)
    campaign=int(campaign)
    # campaign=int(scaler.fit_transform([[campaign]]))

    pdays=st.slider("Enter number of days that passed by after the client was last contacted from a previous campaign",0,27)
    pdays=int(pdays)
    # pdays=int(scaler.fit_transform([[pdays]]))

    previous=st.slider("Enter number of contacts performed before this campaign and for this client",0,7)
    previous=int(previous)
    # previous=int(scaler.fit_transform([[previous]]))

    poutcome=st.radio("Enter outcome of the previous marketing campaign",("select",'failure','nonexistent','success'))

    emp_var_rate= int(1.100)

    cons_price_idx = int(93.749)

    cons_conf_idx = int(-41.8)

    euribor3m = int(4.857)
    

    if marital!="select" and education!="select" and default!="select" and housing!="select" and loan!="select" and contact!="select" and month!="select" and day_of_week!="select" and duration!="sellect" and poutcome!="select":
        marital=int(encoder.fit_transform([[marital]]))
        education=int(encoder.fit_transform([[education]]))
        default=int(encoder.fit_transform([[default]]))
        housing=int(encoder.fit_transform([[housing]]))
        loan=int(encoder.fit_transform([[loan]]))
        contact=int(encoder.fit_transform([[contact]]))
        month=int(encoder.fit_transform([[month]]))
        day_of_week=int(encoder.fit_transform([[day_of_week]]))
        poutcome=int(encoder.fit_transform([[poutcome]]))

        with open("D:\\optimal_model.pkl",'rb') as f:
            rf=pickle.load(f)
        res=rf.predict([[age,job,marital,education,default,housing,loan,contact,month,duration,campaign,pdays,previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m]])
        res=str(res)
        dict={"yes":'1',"no":'0'}    
        for i,j in dict.items():
            res=res.replace(j,i)
    else:
        res="None"


    

if __name__=='__main__':
    Res=main()
    # Res=str(int(result))
    # dict={"yes":'1',"no":'0'}    
    # for i,j in dict.items():
    #     Res=Res.replace(j,i)
    
            
    if st.sidebar.button("Show Prediction"):
        st.sidebar.subheader("The predicted response of customer or client to subscribe a term deposit is")
        st.sidebar.success(Res)
        st.sidebar.header('   ')
        st.sidebar.image('https://media.giphy.com/media/jO2PObgv4R3JrSBFul/giphy.gif')


        
    
    if st.button("Thanks") :
        st.text("Thank you for visiting  and happy learning :)")
        st.balloons()
    
   
# streamlit run Bank_Marketing\bank_marketing_streamlit.py