import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore")
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.metrics import precision_recall_curve
# from sklearn import metrics
from sklearn import tree
# from xgboost import XGBClassifier
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
    
    num_cols = process_value_df[process_value_df['dtype'] != 'object']['feature'].tolist()
    cat_cols = process_value_df[process_value_df['dtype'] == 'object']['feature'].tolist()

    missing_val = 'unknown'
    un_replaced_lst = ['default']
    for col in cat_cols:
        if (col in un_replaced_lst):
            continue

        replaced_val = process_value_df[process_value_df['feature'] == col]['missing_rep_val'].iloc[0]
        client_df[col] = client_df[col].apply(lambda val: replaced_val if val == missing_val else val)


    for col in num_cols:
        if col == 'pdays':
            client_df[col] = client_df[col].map(transform_pdays)
            cat_cols = cat_cols +['pdays'] 
            continue
        else:
          
            replaced_val = process_value_df[process_value_df['feature'] == col]['high_outlier_rep_val'].iloc[0]

            client_df[col] = client_df[col].apply(lambda val: replaced_val if val > replaced_val else val)

    ## onehot_encoding:
    cat_cols = [col for col in client_df.dtypes[client_df.dtypes == 'object'].index.tolist()]  
    # labelencoder = LabelEncoder()
    for column in cat_cols:
        client_df[column] = labelencoder.fit_transform(client_df[column])
        
    return(client_df)  
 
def quick_predict_client(model):
    # Show the input form to get informations
    client_df_ok = pd.read_csv("data/ok_client.csv", index_col = 0)

    target = 'y'
    tam = 2
    
    cols = process_value_df['feature'].tolist()
    col_types = process_value_df['dtype'].tolist()
    
    client_df = pd.DataFrame()
    question_dict = {"age": "Enter age of the customer", "job": "Enter the type of job customer do","marital": "what is customer's marital status?", "education":"Enter customer's education level?",
    "default": "Do the customer have credit in default?", "housing": "Do the customer have Housing Loan?", "loan": "Do the customer have Personal Loan?", "contact": "How do you prefer to communicate",
    "month": "Which month the customer was last contacted in?", "day_of_week": "Which day of week the customer was contacted in", "duration": "Enter last contact duration with the customer in sec?",
    "campaign": "Enter number of contacts performed during this campaign and for this client", "pdays": "Enter number of days that passed by after the client was last contacted from a previous campaign",
    "previous": "Enter number of contacts performed before this campaign and for this client", "poutcome": "Enter outcome of the previous marketing campaign", "emp.var.rate": "Enter employment variation rate - quarterly indicator",
    "cons.price.idx": "Enter consumer price index - monthly indicator", "cons.conf.idx": "Enter consumer confidence index - monthly indicator", "euribor3m": "Enter euribor 3 month rate - daily indicator", "nr.employed": "Enter number of employees - quarterly indicator"}
    for col, col_dtype, key in zip(cols,col_types, question_dict):
        if (col_dtype == 'object'):

            col_option_lst = process_value_df[process_value_df['feature'] == col]['unique_vals'].iloc[0].split(',')
        
            col_selected = col_option_lst.index(client_df_ok.iloc[tam][col])
            col_option= st.selectbox(question_dict[key],options = col_option_lst, index = col_selected)  
            client_df[col] = [col_option]
        else:
            if (col in ['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m']): 
                min_val  = float(process_value_df[process_value_df['feature'] == col]['min'].iloc[0])
                max_val  = float(process_value_df[process_value_df['feature'] == col]['max'].iloc[0])
                value =  float(client_df_ok.iloc[tam][col])
                step = 0.1
            else:
                min_val  = int(process_value_df[process_value_df['feature'] == col]['min'].iloc[0])
                max_val  = int(process_value_df[process_value_df['feature'] == col]['max'].iloc[0])
                value =  int(client_df_ok.iloc[tam][col])
                step = 1

            col_option = st.slider(question_dict[key], min_value= min_val, max_value= max_val, value=value, step=step)
            
            client_df[col] = [col_option]
   
    # Predict 
    if st.button('Show Prediction'):
        st.write(client_df)
        client_df = process_input_client(client_df)        
        X_test = scaler.transform(client_df)  
        y_pred = model.predict(X_test)
        result_str = 'POTENTIAL' if y_pred == 1 else 'NON-POTENTIAL'
        result ='This is a '+ result_str + ' customer for tele-marketing campaign'
        if y_pred == 1:
            st.success(result)
        else:
            st.warning(result)

def visualize_predicted_result(df, target):
    st.subheader("The Predicted Percentage Of Success:")
    data = df.groupby(target).size().sort_values(ascending=False)
    label_dict = {1:'yes',0:'no'}
    fig = plt.figure(figsize = (3,3))
    plt.pie(x=data , autopct='%.1f%%',  labels= [label_dict[val] for val in data.index.tolist()], pctdistance=0.7, radius=1.1)
    col1, col2, col3 = st.beta_columns(3)
    col2.pyplot(fig)   
    
  
def predict_data_file(file,model):
    upload_data = get_df(file)
    features = process_value_df['feature'].tolist()
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
    st.subheader("Choose View:")
    view_result_option = ['View All',"View Successful List", "View Unsuccesful List"]
    col1, col2, col3, col4 = st.beta_columns(4)
    view_type_id = col1.selectbox('',options = view_result_option)
    if (view_type_id == view_result_option[0]):
        st.write(upload_data) 
    else:
        if (view_type_id == view_result_option[1]):
            view_filter = upload_data[result_col] == 1 
        else:
            view_filter = upload_data[result_col] == 0
        # upload_data[view_filter].to_csv("data/successful_client.csv")
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

def view_models_summary(df):
    ## Evaluation metrics
    st.markdown('Evaluation metrics')
    st.write(df)

def visulize_feature_importances(model_importances,model_name):
    st.markdown("""---""")
    st.subheader("Features' weight in models")
    t = model_importances['Weight'].sort_values(ascending = False).index.tolist()
    fig = plt.figure(figsize = (12,9))
    sns.barplot( x = model_importances.iloc[t]['Weight'], y = model_importances.iloc[t]['Feature'])
    plt.title('The feature importances of '+ model_name)
    plt.show()
    col1,col2 = st.beta_columns(2)
    col1.pyplot(fig) 
    col2.dataframe(data=model_importances.iloc[t])
    # st.pyplot(fig) 
    
def visualize_decision_tree(model, features):
    st.markdown("""---""")
    st.subheader('Decision Tree on Banking Tele-marketing dataset')
    fig = plt.figure(figsize=(20,15))
    tree.plot_tree(model,feature_names = features,rounded=True, filled = True);
    st.pyplot(fig) 
    
#### L O A D  Data

# data dùng để processing dữ liệu
process_value_file_path = "data/processing_value_df.csv"
process_value_df = pd.read_csv(process_value_file_path,index_col = 0)

scaler_file_path = "model/pkl_scaler.pkl"
scaler = pickle.load(open(scaler_file_path, 'rb'))

xgboost_clf_file_path = "model/pkl_xgboost_model.pkl"
log_clf_file_path = "model/pkl_log_model.pkl"
tree_clf_file_path = "model/pkl_decisionT_model.pkl"
grboost_clf_file_path = "model/pkl_grboost_model.pkl"

metric_file_path = "model/evaluation_metrics.csv"

labelencoder_file_path = "model/pkl_labelencoder.pkl"
labelencoder = pickle.load(open(labelencoder_file_path, 'rb'))

# Main function
def main():

    # Init model
    xgboost_clf = pickle.load(open(xgboost_clf_file_path, 'rb'))
    log_clf = pickle.load(open(log_clf_file_path, 'rb'))
    tree_clf = pickle.load(open(tree_clf_file_path, 'rb'))
    grboost_clf = pickle.load(open(grboost_clf_file_path, 'rb'))
    
    model_dict = {"XGBoost Classifier" : xgboost_clf
                  ,"GradientBoost Classifier": grboost_clf
                  ,'Decision Tree Classifier': tree_clf
                  ,'Logistic Regressor' : log_clf}


    st.title("Bank Marketing Prediction")
    
    htk=  """
    <div style="background-color:#004d99;padding:0px">
    <h2 style="color:white;text-align:center;">CUSTOMER DESPOIT PREDICTION APP </h2>
    </div>
    """
    st.markdown(htk,unsafe_allow_html=True)
    
    ## Summary models
    st.sidebar.subheader('Predict and Summarize')
    menu_option = ['Make a prediction',"View model summary"]
                    
    menu_type_id = st.sidebar.selectbox('Your choice:',options = menu_option)
    
    ## View summary
    if(menu_type_id == menu_option[1]):
        metric_df = pd.read_csv(metric_file_path)
        ## Evalutation metrics
        view_models_summary(metric_df)
        
        ## Visualize feature importance
        importance_option = [val for val in model_dict.keys()]
        importance_type_id = st.sidebar.radio('View feature importances of',options = importance_option)
      
        model = model_dict[importance_type_id]
        features = [i for i in process_value_df.feature.tolist()]
        
        model_importances = pd.DataFrame({'Feature': features})
        if model == log_clf:
            model_importances['Weight']= model.coef_[0]
        else:
            model_importances['Weight']= model.feature_importances_      
        visulize_feature_importances(model_importances,importance_type_id)  
        
        if(importance_type_id == 'Decision Tree Classifier'):
            visualize_decision_tree(model, features) 
            
        
    else:
    ## Make prediction
        if (menu_type_id == menu_option[0]):
            model = xgboost_clf
            predict_option = ['Quick Predict','Predict With Data File']
            predict_type_id = st.sidebar.radio('Choose predict',options = predict_option)
            
            ## Quick predict
            if (predict_type_id  == predict_option[0]):
                quick_predict_client(model)
                ## Predict on file
            elif (predict_type_id  == predict_option[1]):
                file = st.file_uploader("Upload file", type=['csv'])
                if not file:
                    st.write("Upload a .csv or .xlsx file to get started")
                else:
                    predict_data_file(file,model)
         
   
    if st.sidebar.button("Thanks") :
        st.text("Thank you for visiting  and happy learning :)")
        st.balloons()      
main()

## Run: streamlit run Nguyen_Bank_Marketing_streamlit.py