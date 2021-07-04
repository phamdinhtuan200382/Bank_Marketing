import streamlit as st
import pandas as pd
import numpy as np
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

from sklearn.preprocessing import MinMaxScaler
import json
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from xgboost import XGBClassifier
#import scikitplot as skplt
import matplotlib.pyplot as plt
import base64
st.set_page_config(layout="wide")
def transform_pdays(val):
    transform_dict = {999:'not_previously_contacted',7: 'over_a_week',0:'within_a_week'}
    for key in transform_dict.keys():
        if (val >= key):
            return transform_dict[key]
        
### processing input data
def process_input_client(client_df):
    num_cols = marketing_df.dtypes[marketing_df.dtypes != 'object'].index.tolist()
    cat_cols = [col for col in marketing_df.dtypes[marketing_df.dtypes == 'object'].index.tolist() if col != 'y']

    missing_val = 'unknown'
    un_replaced_lst = ['default']
    for col in cat_cols:
        if (col in un_replaced_lst):
            continue
        replaced_val = marketing_df[col].mode().values.tolist()[0]
        client_df[col] = client_df[col].apply(lambda val: replaced_val if val == missing_val else val)


    for col in num_cols:
        if col == 'pdays':
            client_df[col] = client_df[col].map(transform_pdays)
            cat_cols = cat_cols +['pdays'] 
            continue

        if col == 'campaign':
            replaced_val = 6
        else:
            replaced_val = marketing_df[col].quantile(0.95)   

        client_df[col] = client_df[col].apply(lambda val: replaced_val if val > replaced_val else val)

    ## onehot_encoding:
    cat_cols = [col for col in client_df.dtypes[client_df.dtypes == 'object'].index.tolist()]  
    labelencoder = LabelEncoder()
    for column in cat_cols:
        client_df[column] = labelencoder.fit_transform(client_df[column])
        
    return(client_df)  
 
def get_cat_cols_val(data):
    unique_dict= {}
    cat_cols = data.dtypes[data.dtypes == 'object'].index
    for col in cat_cols:
        unique_col = data[col].unique()
        unique_dict[col] = unique_col
    return unique_dict

def quick_predict_client():
    client_df_ok = pd.read_csv("data/X_client_df.csv", index_col = 'Unnamed: 0')

    target = 'y'
    tam = 1
    # num_cols = marketing_df.dtypes[marketing_df.dtypes != 'object'].index.tolist()
    # cat_cols = [col for col in marketing_df.dtypes[marketing_df.dtypes == 'object'].index.tolist() if col != target]
    cols = [col for col in marketing_df.columns.tolist() if col != 'y']
    col_types = [marketing_df[col].dtype.name for col in cols]
    client_df = pd.DataFrame()
    question_dict = {"age": "Enter age of the customer", "job": "Enter the type of job customer do","marital": "what is customer's marital status?", "education":"Enter customer's education level?",
    "default": "Do the customer have credit in default?", "housing": "Do the customer have Housing Loan?", "loan": "Do the customer have Personal Loan?", "contact": "How do you prefer to communicate",
    "month": "Which month the customer was last contacted in?", "day_of_week": "Which day of week the customer was contacted in", "duration": "Enter last contact duration with the customer in sec?",
    "campaign": "Enter number of contacts performed during this campaign and for this client", "pdays": "Enter number of days that passed by after the client was last contacted from a previous campaign",
    "previous": "Enter number of contacts performed before this campaign and for this client", "poutcome": "Enter outcome of the previous marketing campaign", "emp.var.rate": "Enter employment variation rate - quarterly indicator",
    "cons.price.idx": "Enter consumer price index - monthly indicator", "cons.conf.idx": "Enter consumer confidence index - monthly indicator", "euribor3m": "Enter euribor 3 month rate - daily indicator", "nr.employed": "Enter number of employees - quarterly indicator"}
    for col, col_dtype, key in zip(cols,col_types, question_dict):
        if (col_dtype == 'object'):
            col_option_lst = marketing_df[col].unique().tolist()
            col_mode = marketing_df[col].mode().tolist()[0]
            
            col_selected = col_option_lst.index(client_df_ok.iloc[tam][col])
            col_option= st.selectbox(question_dict[key],options = col_option_lst, index = col_selected)  
            client_df[col] = [col_option]
        else:
            min_val  = int(marketing_df[col].min())
            max_val  = int(marketing_df[col].max())
            value =  int(client_df_ok.iloc[tam][col])
            col_option = st.slider(question_dict[key], min_value= min_val, max_value= max_val, value=value, step=1)
            
            client_df[col] = [col_option]

    if st.button('Show Prediction'):
            st.write(client_df)
            #client_df = client_df_ok .drop(['y'], axis = 1) 
            client_df = process_input_client(client_df)
            X_test = scaler.transform(client_df)
            
            y_pred = model.predict(X_test)
            result_str = 'POTENTIAL' if y_pred == 1 else 'NON-POTENTIAL'
            result ='This is a '+ result_str + ' client for this tele marketing campain'
            st.success(result)

def visualize_predicted_result(df, target):
    st.subheader("The Predicted Percentage Of Success:");
    data = df.groupby(target).size().sort_values(ascending=False)
    label_dict = {1:'yes',0:'no'}
    fig = plt.figure()
    plt.pie(x=data , autopct="%.1f%%",pctdistance=0.5, labels= [label_dict[val] for val in data.index.tolist()], radius=0.5);
    st.pyplot(fig)  
    
def predict_data_file(file):
    upload_data = get_df(file)
    features = [col for col in marketing_df.columns.tolist() if col != 'y']
    input_data  = upload_data[features]
    st.subheader('List Of Customers To Predict:')
    st.write(input_data)
    
    ## prediction
    client_df = process_input_client(input_data)
    X_client_test = scaler.transform(client_df)
    y_client_pred = model.predict(X_client_test)
    result_col = 'predict'
    upload_data[result_col] = y_client_pred
    
    ## summary result  
    pred_success_cnt = sum((y_client_pred == 1))
    total_cnt = len(y_client_pred)
    st.subheader('Predicted Result:')
    st.success (str(pred_success_cnt) +" customers say YES over "+str(total_cnt) +' people')
    visualize_predicted_result(upload_data, result_col)
    
    ## view result
    view_result_option = ['View All',"View Successful List", "View Unsuccesful List"]
    st.subheader("Choose View:")
    view_type_id = st.selectbox('',options = view_result_option)
    if (view_type_id == view_result_option[0]):
        st.write(upload_data) 
    else:
        if (view_type_id == view_result_option[1]):
            view_filter = upload_data[result_col] == 1 
        else:
            view_filter = upload_data[result_col] == 0
        
        st.write(upload_data[view_filter])
     

     
        
# Initial setup
# st.set_page_config(layout="wide")
def get_df(file):
      # get extension and read file
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(file,sep = ',')
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(file, engine='openpyxl')
  elif extension.upper() == 'PICKLE':
    df = pd.read_pickle(file)
  return df

#### L O A D  Data
data_file_path = "data/bank-additional-full.csv"
marketing_df = pd.read_csv(data_file_path,sep = ";")

model_file_path = "model/pkl_model.pkl"
model = pickle.load(open(model_file_path, 'rb'))

scaler_file_path = "model/pkl_scaler.pkl"
scaler = pickle.load(open(scaler_file_path, 'rb'))

# Main function
def main():
#     main_bg = "image/photo.jfif"
#     main_bg_ext = "jfif"
#     st.markdown(
#     f"""
#     <style>
#     .reportview-container {{
#         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#     }}
#     """,
#     unsafe_allow_html=True
# )
    st.title("Bank Marketing Prediction")
    
    htk=  """
    <div style="background-color:#004d99;padding:0px">
    <h2 style="color:white;text-align:center;">CUSTOMER DESPOIT PREDICTION APP </h2>
    </div>
    """
    st.markdown(htk,unsafe_allow_html=True)
    predict_option = ['Quick Predict','Predict With Data File']
    predict_type_id = st.sidebar.selectbox('CHOOSE PREDICT',options = predict_option)
    
    if (predict_type_id  == predict_option[0]):
        quick_predict_client()
    elif (predict_type_id  == predict_option[1]):
        file = st.file_uploader("Upload file", type=['csv'])
        if not file:
            st.write("Upload a .csv or .xlsx file to get started")
        else:
            predict_data_file(file)
    if st.sidebar.button("Thanks") :
        st.text("Thank you for visiting  and happy learning :)")
        st.balloons()      
main()

## Run: streamlit run apps/bank_marketing_streamlit.py