from re import T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pickle
from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
from PIL import Image
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

st.set_page_config(layout="wide")
# Load data
file_path = "data/bank-additional-full.csv"
na_lst = ["NA","","#NA","unknown"]
marketing_df = pd.read_csv(file_path, sep=';')
marketing_null = pd.read_csv(file_path, sep=';', na_values = na_lst, keep_default_na = True) 
# Def Visualization
na_lst = ["NA","","#NA","unknown"]
def missing_exploration(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = round((data.isnull().sum()/data.isnull().count()*100),2).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent %'])
    return missing_data
def visualize_numerical(df, column, target = 'y'):
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,4)) # Create 2 charts in 1 row
    
    sns.histplot(df[column], ax=ax1, kde=True);
    ax1.set_xlabel(column);
    ax1.set_ylabel('Density');
    ax1.set_title(f'{column}  Distribution');

    sns.boxplot(x=target, y=column, data=df, showmeans=True, ax=ax2);
    ax2.set_xlabel('Target');
    ax2.set_ylabel(column);
    plt.show()

def visualize_numerical_lst(df, numerical = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',\
                            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], target = 'y'):
    for column in numerical:
        visualize_numerical(df,column)
        print();
        
def visualize_categorical(df, column, target = 'y'):
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,4)) 
        data1 = df.groupby(column).size()
        ax1.pie(x=data1 , autopct="%.2f%%",textprops=dict(color='black'), explode=[0.05]*len(data1) , labels=data1.index.tolist(),      pctdistance=0.7, radius=1.1,  startangle=90)
        ax1.set_title(f'{column}  Distribution', loc='center')

        data2 = get_col_target(column, target,df)   
        data2.plot(kind='bar',stacked = True, ax=ax2);
        plt.xticks(rotation=45);        
        plt.show()

def visualize_categorical_w_success_percent(df, column, target = 'y'):
    
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,4)) 
        data1 = df[df['y']=='yes'].groupby(column).size()
        ax1.pie(x=data1 , autopct="%.2f%%",textprops=dict(color='black'), explode=[0.05]*len(data1) , labels=data1.index.tolist(),      pctdistance=0.7, radius=1.1,  startangle=90)
        ax1.set_title(f'Distribution of each field in {column} on total success rate', loc='center')

        data2 = get_col_target(column, target,df)   
        data2.plot(kind='bar',stacked = True, ax=ax2);
        ax2.set_title(f' Quantity distribution by {column}')
        plt.xticks(rotation=45);        
        plt.show()

def visualize_categorical_w_success(df, column, target = 'y'):
    
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,4))    
        #chart number
        data2 = get_col_target(column, target,df)   
        ax1 = data2.plot(kind='bar',stacked = True, ax=ax1);
        ax1.set_title(f' Quantity distribution by {column}', loc='center')
    
        #chart successing rate
        data = get_col_target(column, target,df)
        data['yes_rate'] = round(data['yes']*100/(data['yes'] + data['no']),2)
        yes_rate_df =  data['yes_rate'].sort_values(ascending = False)

        #Rotation
        sns.barplot(y = yes_rate_df.values, x = yes_rate_df.index, ax=ax2)
        ax2.set_title(f' Successing rate by {column}')
        for ax in fig.axes:
            ax.tick_params(labelrotation=45)
    
def visualize_categorical_lst(df,categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',\
                                          'month', 'day_of_week', 'poutcome'], target = 'y'):
    for column in categorical:
        visualize_categorical(df, column)
        
def visualize_success_rate(df,categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',\
                                          'month', 'day_of_week', 'poutcome'], target = 'y'):
    
    for column in categorical:
        data = get_col_target(column, target,df)
        data['yes_rate'] = round(data['yes']*100/(data['yes'] + data['no']),2)
        yes_rate_df =  data['yes_rate'].sort_values(ascending = False)


        plt.figure(figsize = (8,5))
        sns.barplot(y = yes_rate_df.values, x = yes_rate_df.index)
        plt.xticks(rotation = 45)
        plt.ylabel('Successful rate by '+ column)
        
def get_col_target(rows, cols,data):
    
    cols_lst = data[cols].unique().tolist()
    rows_lst = data.groupby(rows)[rows].count().sort_values(ascending = False).index.tolist()

    group_df = data.groupby([rows,cols]).size()
    dic = {}
    for item in cols_lst:
        vals = []
        for i in rows_lst:
            try:
                vals.append(group_df.loc[(i, item)])
            except:
                vals.append(0)
            finally:
                continue
        dic[item] = vals

    df = pd.DataFrame(dic,index = rows_lst)
    return(df)

# Text describe
general = """
        <p style="margin: 0in; font-family: Calibri, sans-serif; line-height: 1.5;"><span style="font-size: 14px; font-family: Calibri, sans-serif;">Nhận x&eacute;t:</span></p>
        <ul style="list-style-type: square;">
        <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif; line-height: 1.5;"><span style="font-size: 14px;"><span style="font-family: Calibri, sans-serif;">Bộ dữ liệu c&oacute; 41188 d&ograve;ng v&agrave; 21 cột, trong đ&oacute; c&oacute; 10 biến số v&agrave; 11 biến ph&acirc;n loại.&nbsp;</span></span></li>
        <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif; line-height: 1.5;"><span style="font-size: 14px; font-family: Calibri, sans-serif;">Bộ dữ liệu c&oacute; 88.7% nh&atilde;n &apos;no&apos; v&agrave; 11.3% nh&atilde;n &apos;yes&apos;, điều n&agrave;y chỉ ra rằng bộ dữ liệu nghi&ecirc;n cứu kh&ocirc;ng c&acirc;n bằng giữa tỉ lệ c&aacute;c kết quả thu được. N&oacute; đồng thời cũng cho thấy rằng ng&acirc;n h&agrave;ng n&agrave;y đ&atilde; thực hiện chiến dịch call marketing n&agrave;y một c&aacute;ch kh&ocirc;ng t&iacute;nh to&aacute;n n&ecirc;n tỉ lệ th&agrave;nh c&ocirc;ng thật sự thấp.</span></li>
        </ul>
        """
age = """
            <p style="line-height: 1.15;"><span style="font-family: Calibri, sans-serif; font-size: 14px;">Nhận x&eacute;t:&nbsp;</span></p>
            <ul style="list-style-type: square;">
            <li style="line-height: 1.15;"><span style="font-size: 14px;"><span style="font-family: Calibri, sans-serif;">Trong chiến dịch n&agrave;y ng&acirc;n h&agrave;ng tập trung v&agrave;o nh&oacute;m đối tượng từ 25-60 tuổi, nh&oacute;m tuổi target trong chiến dịch n&agrave;y l&agrave; từ 30-40 tuổi.&nbsp;</span></span></li>
            <li style="line-height: 1.15;"><span style="font-size: 14px;"><span style="font-family: Calibri, sans-serif;">Những kh&aacute;ch h&agrave;ng đồng &yacute; gởi tiền tập trung ở độ tuổi 30-50. Những người đồng &yacute; gởi tiền mặc d&ugrave; bi&ecirc;n động rộng hơn những người từ chối, nhưng nh&igrave;n chung họ c&oacute; độ tuổi trung b&igrave;nh trẻ hơn những người kh&ocirc;ng đồng &yacute;.&nbsp;</span></span></li>
            </ul>
            <p style="line-height: 1.15;"><span style="font-family: Calibri, sans-serif; font-size: 14px;">Lưu &yacute;: Biến age c&oacute; phần outlier từ khoảng 70 tuổi trở l&ecirc;n, n&ecirc;n c&oacute; thể c&acirc;n nhắc loại bỏ c&aacute;c gi&aacute; trị n&agrave;y trong phần model training.</span></p>
             """
job = """<p style="margin: 0in;font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Chiến dịch n&agrave;y tập trung v&agrave;o gọi điện cho đối tượng ch&iacute;nh l&agrave; admin, người lao động ch&acirc;n tay v&agrave; những người l&agrave;m kĩ thuật l&agrave; ch&iacute;nh.</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Mặc d&ugrave; 3 nh&oacute;m m&agrave; chiến dịch tập trung v&agrave;o ch&iacute;nh chiếm số lượng th&agrave;nh c&ocirc;ng gửi tiền cao nhất, tỉ lệ th&agrave;nh c&ocirc;ng chỉ nằm ở mức thấp.</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Th&agrave;nh phần sinh vi&ecirc;n v&agrave; người nghỉ hưu tuy chiếm số lượng kh&ocirc;ng nhiều nhưng lại c&oacute; tỉ lệ th&agrave;nh c&ocirc;ng cao vượt trội so với ng&agrave;nh nghề kh&aacute;c.</span></li>
</ul>
"""
marital = """
<p style="margin: 0in;font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Đối tượng ch&iacute;nh trong chiến dịch n&agrave;y đa số l&agrave; những người đ&atilde; lập gia đ&igrave;nh.</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Mặc d&ugrave; t&iacute;nh về số lượng th&agrave;nh c&ocirc;ng th&igrave; những người kết h&ocirc;n c&oacute; số lượng k&iacute; gửi tiền nhiều nhất, tuy nhi&ecirc;n x&eacute;t về tỉ lệ th&agrave;nh c&ocirc;ng th&igrave; những người thuộc nh&oacute;m độc th&acirc;n chiếm tỉ lệ cao hơn.</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Số lượng &iacute;t data l&agrave; unknown, tuy nhi&ecirc;n lại c&oacute; tỉ lệ th&agrave;nh c&ocirc;ng cao nhất -&gt; c&oacute; khả năng dễ bị bias khi train model nếu train với dữ liệu n&agrave;y.</span></li>
</ul>
"""

education = """
<p style="margin: 0in;font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Trong chiến dịch marketing lần n&agrave;y, kh&aacute;ch h&agrave;ng tập trung ch&iacute;nh v&agrave;o đối tượng c&oacute; bằng cấp đại học v&agrave; cấp ba l&agrave; ch&iacute;nh.</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Đồng thời khi x&eacute;t về tỉ lệ kh&aacute;ch h&agrave;ng gửi tiền, nh&oacute;m kh&aacute;ch h&agrave;ng c&oacute; tr&igrave;nh độ đại học, học sinh cấp ba v&agrave; professional course l&agrave; ba nh&oacute;m chiếm tỉ lệ cao nhất.</span></li>
</ul>
"""

default = """
<p style="margin: 0in;font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Trong chiến dịch n&agrave;y, ng&acirc;n h&agrave;ng tập trung v&agrave;o đối tượng ch&iacute;nh l&agrave; những người kh&ocirc;ng c&oacute; nợ xấu</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Yếu tố default l&agrave; yếu tố c&oacute; độ inbalance lớn đồng thời c&oacute; số lượng missing gần 10.000</span></li>
</ul>
"""
housing = """
<p style="margin: 0in;font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Trong chiến dịch n&agrave;y, x&eacute;t về những kh&aacute;ch h&agrave;ng m&agrave; chiến dịch li&ecirc;n hệ c&oacute; tỉ lệ ch&ecirc;nh lệch giữa những người c&oacute; nh&agrave; m&agrave; kh&ocirc;ng c&oacute; nh&agrave; kh&ocirc;ng qu&aacute; nhiều.</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Theo biểu đồ th&igrave; tỉ lệ kh&aacute;ch h&agrave;ng c&oacute; nh&agrave; gửi tiền cao hơn tuy nhi&ecirc;n độ ch&ecirc;nh lệch kh&ocirc;ng đ&aacute;ng kể</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Biến n&agrave;y c&oacute; số lượng nhỏ missing value, đồng thời tỉ lệ th&agrave;nh c&ocirc;ng tr&ecirc;n missing value bằng với tỉ lệ th&agrave;nh c&ocirc;ng của No -&gt; c&oacute; thể c&acirc;n nhắc phương ph&aacute;p xử l&iacute; ph&ugrave; hợp để tr&aacute;nh việc bias.</span></li>
</ul>
"""

loan = """
<p style="margin: 0in;font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Trong chiến dịch n&agrave;y, ng&acirc;n h&agrave;ng tập trung chủ yếu v&agrave;o những đối tượng kh&ocirc;ng c&oacute; nợ -&gt; biến c&oacute; độ inbalance cao n&ecirc;n cần c&acirc;n nhắc th&ecirc;m về mối li&ecirc;n hệ với c&aacute;c biến kh&aacute;c.</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif;"><span style="font-size: 14px;">Biến nợ c&oacute; một số lượng missing value kh&ocirc;ng đ&aacute;ng kể.</span></li>
</ul>
"""

contact = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:&nbsp;</span></p>
<ul style="list-style-type: square;">
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Trong chiến dịch n&agrave;y, ng&acirc;n h&agrave;ng chọn phương ph&aacute;p gọi điện bằng điện thoại để li&ecirc;n hệ với kh&aacute;ch h&agrave;ng. Trong đ&oacute; số lượng kh&aacute;ch h&agrave;ng được li&ecirc;n hệ bằng di động chiếm số đ&ocirc;ng so với việc gọi điện bằng điện thoại b&agrave;n.</span></li>
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Tỉ lệ th&agrave;nh c&ocirc;ng đối với phương thức gọi điện cũng chiếm tỉ lệ cao gần gấp 3 lần so với việc gọi điện bằng số điện thoại b&agrave;n chứng tỏ việc tiếp cận kh&aacute;ch h&agrave;ng bằng c&aacute;ch gọi điện bằng điện thoại di động c&oacute; hiệu quả hơn nhiều so với việc gọi điện bằng số điện thoại b&agrave;n.</span></li>
</ul>
"""

day_of_week = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Trong chiến dịch n&agrave;y, ng&acirc;n h&agrave;ng gọi điện hầu như ph&acirc;n bổ đều c&aacute;c ng&agrave;y trong tuần.&nbsp;</span></li>
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Tuy nhi&ecirc;n t&iacute;nh tr&ecirc;n tỉ lệ th&agrave;nh c&ocirc;ng th&igrave; ng&agrave;y đầu tuần (thứ hai) v&agrave; ng&agrave;y cuối tuần (thứ s&aacute;u) c&oacute; tỉ lệ th&agrave;nh c&ocirc;ng thấp nhất.</span></li>
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Chiến dịch c&oacute; thể c&acirc;n nhắc li&ecirc;n hệ kh&aacute;ch h&agrave;ng v&agrave;o c&aacute;c ng&agrave;y giữa tuần từ thứ ba đến thứ năm để c&oacute; hiệu quả cao hơn.</span></li>
</ul>
"""

duration = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Thời gian gọi điện trung b&igrave;nh cho kh&aacute;ch h&agrave;ng nằm ở 449s</span></li>
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Thời gian của cuộc gọi gần nhất (duration) tập trung trong khoảng 0-200.&nbsp;</span></li>
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Những người c&oacute; duration = 0 th&igrave; tỉ lệ th&agrave;nh c&ocirc;ng gần như l&agrave; 0. Ở những trường hợp tiếp cận th&agrave;nh c&ocirc;ng, thời lượng trung b&igrave;nh của cuộc gọi gần nhất trong chiến dịch hiện tại khoảng 450 (tập trung trong khoảng &lt; 1000)</span></li>
</ul>
"""

campaign = """
<p style="margin-bottom: 10px !important; caret-color: rgb(0, 0, 0); color: rgb(0, 0, 0); font-family: -webkit-standard; font-style: normal; font-variant-caps: normal; font-weight: normal; letter-spacing: normal; orphans: auto; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: auto; word-spacing: 0px; -webkit-text-size-adjust: auto; -webkit-text-stroke-width: 0px; text-decoration: none;"><span style="font-size: 14px;">Nhận x&eacute;t:&nbsp;</span></p>
<ul style="list-style-type: square;">
    <li style="margin-bottom: 10px !important; caret-color: rgb(0, 0, 0); color: rgb(0, 0, 0); font-family: -webkit-standard; font-style: normal; font-variant-caps: normal; font-weight: normal; letter-spacing: normal; orphans: auto; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: auto; word-spacing: 0px; -webkit-text-size-adjust: auto; -webkit-text-stroke-width: 0px; text-decoration: none;"><span style="font-size: 14px;">Trong chiến dịch n&agrave;y, hầu hết c&aacute;c kh&aacute;ch h&agrave;ng được li&ecirc;n hệ từ 1-3 lần v&agrave; tỉ lệ kh&aacute;ch h&agrave;ng đồng &yacute; gửi tiền cũng chiếm số đ&ocirc;ng trong 3 lần li&ecirc;n hệ trở lại.</span></li>
</ul>
<p style="margin: 0in; caret-color: rgb(0, 0, 0); color: rgb(0, 0, 0); font-style: normal; font-variant-caps: normal; font-weight: normal; letter-spacing: normal; orphans: auto; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: auto; word-spacing: 0px; -webkit-text-size-adjust: auto; -webkit-text-stroke-width: 0px; text-decoration: none; font-family: Calibri, sans-serif; line-height: 1.5;"><span style="font-size: 14px;">&nbsp;</span></p>
"""

pdays = """<p style="margin: 0in; font-family: Calibri, sans-serif; line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif; line-height: 1.5;"><span style="font-size: 14px;">Đa số chưa được tiếp x&uacute;c qua điện thoại ở chiến dịch quảng c&aacute;o trc đ&oacute; (pdays = 999).&nbsp;</span></li>
    <li style="margin-top: 0in; margin-right: 0in; margin-bottom: 0in; font-family: Calibri, sans-serif; line-height: 1.5;"><span style="font-size: 14px;">Đối với những clients đ&atilde; được tiếp cận ở chiến dịch trước (pdays != 999), đa số cuộc gọi cuối c&ugrave;ng của chiến dịch trước nằm trong khoảng 3 - 6 ng&agrave;y (trong v&ograve;ng 1 tuần). Trong đ&oacute;, những trường hợp th&agrave;nh c&ocirc;ng c&oacute; pdays trung b&igrave;nh l&agrave; 5 ng&agrave;y.</span></li>
</ul>
<p style="margin: 0in; font-family: Calibri, sans-serif; line-height: 1.5;"><span style="font-size: 14px;">** Note: Data thể hiện tr&ecirc;n biểu đồ c&oacute; pre-process loại bỏ c&aacute;c loại bỏ c&aacute;c gi&aacute; trị &apos;999&apos; của pdays </span></p>
"""

previous = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Số cuộc gọi đến kh&aacute;ch h&agrave;ng thực hiện trước chiến dịch hiện tại (previous) phần nhiều nằm ở 0 --&gt; Đa số kh&aacute;ch h&agrave;ng ở chiến dịch lần n&agrave;y l&agrave; mới, chưa đc tiếp thị lần n&agrave;o.&nbsp;</span></li>
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Trung b&igrave;nh số lần gọi đến kh&aacute;ch h&agrave;ng trước chiến dịch hiện tại ở những người đồng &yacute; cao hơn ở nh&oacute;m người kh&ocirc;ng đồng &yacute;, lần lượt l&agrave; 0.5 v&agrave; 0. C&oacute; thể n&oacute;i sự tương t&aacute;c qua điện thoại đến kh&aacute;ch h&agrave;ng trước đ&oacute; cũng ảnh hưởng đến kết quả của chiến dịch ở lần tiếp theo. (Nếu trước đ&oacute;, kh&aacute;ch h&agrave;ng đ&atilde; được chăm s&oacute;c, họ c&oacute; xu hướng mua sản phẩm sẽ được tiếp thị)</span></li>
</ul>
"""

poutcome = """
<p><span style="font-size: 14px;">Nhận x&eacute;t:</span></p>
<ul style="list-style-type: square;">
    <li><span style="font-size: 14px;">Trong chiến dịch n&agrave;y hầu hết c&aacute;c li&ecirc;n hệ đều l&agrave; kh&aacute;ch h&agrave;ng mới.</span></li>
    <li><span style="font-size: 14px;">Nếu ng&acirc;n h&agrave;ng đ&atilde; th&agrave;nh c&ocirc;ng trong việc thuyết phục kh&aacute;ch h&agrave;ng sử dụng dịch vụ trước đ&oacute; th&igrave; tỉ lệ kh&aacute;ch h&agrave;ng đồng &yacute; tham gia chiến dịch lần tới sẽ cao, l&ecirc;n tới tr&ecirc;n 60%</span></li>
</ul>
"""

emp_var_rate = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:&nbsp;</span></p>
<ul style="list-style-type: square;">
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Hệ số em.var.rate (hệ số thay đổi c&ocirc;ng việc)</span></li>
    <li style="line-height: 1.5;"><span style="font-size: 14px;">KH được tiếp thị c&oacute; tỉ lệ thay đổi c&ocirc;ng việc nằm từ -2 đến 1.2. Đặc biệt, những KH đồng &yacute; gởi tiền đều c&oacute; tỉ lệ n&agrave;y &lt;0 (hiếm khi thay đổi c&ocirc;ng việc) v&agrave; tỉ lệ n&agrave;y thấp hơn tỉ lệ trung b&igrave;nh của những người kh&ocirc;ng đồng &yacute; gởi. Những người &iacute;t thay đổi c&ocirc;ng việc c&oacute; xu hướng gởi tiền ng&acirc;n h&agrave;ng nhiều hơn người hay thay đổi c&ocirc;ng việc.</span></li>
</ul>
"""

cons_price_idx = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">Hệ số cons.price.idx l&agrave; hệ số gi&aacute; ti&ecirc;u d&ugrave;ng.</li>
<li style="line-height: 1.5;"><span style="font-size: 14px;">Hệ số gi&aacute; ti&ecirc;u d&ugrave;ng của kh&aacute;ch h&agrave;ng được gọi nằm trong khoảng 93-94. Tỉ lệ trung b&igrave;nh của chỉ số n&agrave;y ở những người đồng &yacute; gởi thấp hơn những người ko gởi. Điều n&agrave;y c&oacute; nghĩa l&agrave; khi chỉ số gi&aacute; ti&ecirc;u d&ugrave;ng cao (gi&aacute; trị h&agrave;ng h&oacute;a b&aacute;n ra cao), kh&aacute;ch h&agrave;ng sẽ &iacute;t c&oacute; xu hướng gởi tiền.</li>
</ul>
"""

cons_conf_idx = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">Hệ số cons.conf.ind (chỉ số lạc quan thị trường)</li>
<li style="line-height: 1.5;"><span style="font-size: 14px;">Tỉ lệ trung b&igrave;nh ở những kh&aacute;ch h&agrave;ng tiếp thị th&agrave;nh c&ocirc;ng lại cao hơn. Tỉ lệ n&agrave;y nằm trong khoảng từ -45 đến -37. Điều n&agrave;y c&oacute; nghĩa Khi kh&aacute;ch h&agrave;ng tin tưởng v&agrave;o sức khỏe của nền kinh tế th&igrave; người ta cũng c&oacute; xu hướng gởi tiền v&agrave;o ng&acirc;n h&agrave;ng nhiều hơn.</li>
</ul>
"""

euribor3m = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">Euribo3m (l&agrave; tỉ lệ tham chiếu được x&acirc;y dựng từ l&atilde;i suất trung b&igrave;nh m&agrave; c&aacute;c ng&acirc;n h&agrave;ng Ch&acirc;u &Acirc;u cung cấp cho vay ngắn hạn kh&ocirc;ng c&oacute; t&agrave;i sản bảo đảm tr&ecirc;n thị trường li&ecirc;n ng&acirc;n h&agrave;ng): nằm trong khoảng từ 1-5.</li>
<li style="line-height: 1.5;"><span style="font-size: 14px;">So s&aacute;nh 2 nh&oacute;m đồng &yacute; gởi v&agrave; kh&ocirc;ng gởi, ta thấy trung b&igrave;nh chỉ số n&agrave;y ở nh&oacute;m người đồng &yacute; thấp hơn nh&oacute;m kh&ocirc;ng đồng &yacute;.Điều n&agrave;y c&oacute; nghĩa khi chỉ số euribo3m c&agrave;ng thấp th&igrave; c&agrave;ng c&oacute; nhiều người gởi hơn.</li>
<li style="line-height: 1.5;"><span style="font-size: 14px;">Số lượng người c&oacute; việc l&agrave;m t&iacute;nh theo qu&yacute; nằm trong khoảng 5010 - 5210. Khi chỉ số n&agrave;y c&agrave;ng thấp th&igrave; client c&agrave;ng c&oacute; xu hướng gởi hơn.</li>
</ul>
"""

nr_employed = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Nhận x&eacute;t:</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">Biến nr.employed l&agrave; biến x&atilde; hội thể hiện số lượng người c&oacute; việc l&agrave;m t&iacute;nh tr&ecirc;n qu&yacute;.</li>
<li style="line-height: 1.5;"><span style="font-size: 14px;">C&oacute; thể nhận thấy l&agrave; số lượng người c&oacute; việc l&agrave;m nằm ở 5025-5200 th&igrave; c&oacute; khả năng th&agrave;nh c&ocirc;ng cao trong chiến dịch, đồng thời chỉ số n&agrave;y nằm ở 5100 l&agrave; chỉ số l&iacute; tưởng nhất để tiếp cận kh&aacute;ch h&agrave;ng.</li>
</ul>
"""

comment_1 = """
<pli style="line-height: 1.5;"><span style="font-size: 14px;">### Những ch&uacute; &yacute; cho qu&aacute; tr&igrave;nh xử l&yacute;</p>
<p style="line-height: 1.5;"><span style="font-size: 14px;">1. Lược bỏ dữ liệu</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">Cột 'duration' sẽ kh&ocirc;ng được quan t&acirc;m trong qu&aacute; tr&igrave;nh ph&acirc;n t&iacute;ch v&agrave; xử l&yacute;</li>
<li style="line-height: 1.5;"><span style="font-size: 14px;">C&oacute; 12 d&ograve;ng tr&ugrave;ng nhau sẽ cắt bỏ</li>
</ul>
<p style="line-height: 1.5;"><span style="font-size: 14px;">2. Missing:</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">C&oacute; sự missing value tr&ecirc;n c&aacute;c biến: default, education, loan, housing, marital, job. Trong đ&oacute; biến default c&oacute; tỉ lệ missing value cao v&agrave; đ&aacute;ng kể nhất (~20%)</li>
</ul>
<p style="line-height: 1.5;"><span style="font-size: 14px;">3. Outlier:</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">Age, campain, previous, cons.conf.idx</li>
</ul>
<p style="line-height: 1.5;"><span style="font-size: 14px;">4. C&aacute;c biến số c&acirc;n nhắc ph&acirc;n loại th&ecirc;m: age, pdays.</p>
<p style="line-height: 1.5;"><span style="font-size: 14px;">5. Clustering theo 3 chỉ số pdays, previous, poutcome</p>
<p>&nbsp;</p>
"""

comment_2 = """
<pli style="line-height: 1.5;"><span style="font-size: 14px;">### Lưu &yacute; kh&aacute;c nh&oacute;m ghi nhận được trong qu&aacute; tr&igrave;nh ph&acirc;n t&iacute;ch ngo&agravei</p>
<p style="line-height: 1.5;"><span style="font-size: 14px;">1. Biến previous:</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">Biến n&agrave;y c&oacute; tương quan tỉ lệ nghịch với nh&oacute;m biến ở mục 1, tuy nhi&ecirc;n những hệ số n&agrave;y kh&ocirc;ng qu&aacute; quan trọng trong phạm vi xem x&eacute;t</li>
</ul>
<p style="line-height: 1.5;"><span style="font-size: 14px;">2.&nbsp;Nh&oacute;m biến x&atilde; hội:</p>
<ul style="list-style-type: square;">
<li style="line-height: 1.5;"><span style="font-size: 14px;">Nh&oacute;m chỉ số nr.employed, emp.var.rate, euribor3m l&agrave; nh&oacute;m c&oacute; tương quan tỉ lệ thuận<br />&gt; 3 biến n&agrave;y c&oacute; yếu tố tương đồng rất cao n&ecirc;n ta c&oacute; thể chọn 1 biến để quan s&aacute;t (để giảm chiều dữ liệu)</li>
<li style="line-height: 1.5;"><span style="font-size: 14px;">3 biến nr.employed, emp.var.rate, euribo3m l&agrave; 3 biến c&oacute; tương quan rất mạnh --&gt; c&oacute; thể lọc bớt để giảm chiều dữ liệu</li>
</ul>
<p style="line-height: 1.5;"><span style="font-size: 14px;">4. Tỉ lệ chuyển đổi: Tỉ lệ chuyển đổi từ Failure to YES l&agrave; 14% trong khi tỉ lệ chuyển đổi từ Success to NO gần 35%. Ở đ&acirc;y c&oacute; sự mất kh&aacute;ch h&agrave;ng.</p>
<p>&nbsp;</p>
"""
pre_process_1 = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">C&aacute;c bước pre-processing được sử dụng:</span></p>
<p style="line-height: 1.5;"><span style="font-size: 14px;">1. Dữ liệu tr&ugrave;ng: Xo&aacute; 12 d&ograve;ng dữ liệu tr&ugrave;ng nhau</span></p>
<p style="line-height: 1.5;"><span style="font-size: 14px;">2. Số lượng biến sử dụng cho training: 19 biến/ 21 biến (loại bỏ 2 cột duration v&agrave; nr.employed)</span></p>
<p style="line-height: 1.5;"><span style="font-size: 14px;">3. Dữ liệu thiếu:</span></p>
<ul style="list-style-type: square;">
    <li><span style="font-size: 14px;">Đối với c&aacute;c biến missing thấp: V&igrave; mục ti&ecirc;u l&agrave; tiếp thị được nhiều kh&aacute;ch h&agrave;ng kh&aacute;ch h&agrave;ng c&agrave;ng tốt - tr&aacute;nh việc đ&aacute;nh mất kh&aacute;ch h&agrave;ng tiềm năng, n&ecirc;n nh&oacute;m quyết định biến đổi c&aacute;c biến missing c&oacute; tỉ lệ nhỏ sang gi&aacute; trị c&oacute; tỉ lệ th&agrave;nh c&ocirc;ng cao nhất trong bộ dữ liệu. Cụ thể:</span>
        <ul>
            <li style="line-height: 1.5;"><span style="font-size: 14px;">Biến education: &apos;unknown&apos; -&gt; &apos;<span style="color: rgb(184, 49, 47);">university-degree</span>&apos;&nbsp;</span></li>
            <li style="line-height: 1.5;"><span style="font-size: 14px;">Biến loan: &apos;unknown&apos; -&gt; &apos;<span style="color: rgb(184, 49, 47);">no</span>&apos;&nbsp;</span></li>
            <li style="line-height: 1.5;"><span style="font-size: 14px;">Biến housing: &apos;unknown&apos; -&gt; &apos;<span style="color: rgb(184, 49, 47);">yes</span>&apos;</span></li>
            <li style="line-height: 1.5;"><span style="font-size: 14px;">Biến marital: &apos;unknown&apos; -&gt; &apos;<span style="color: rgb(184, 49, 47);">single</span>&apos;</span></li>
            <li style="line-height: 1.5;"><span style="font-size: 14px;">Biến job: &apos;unknown&apos; -&gt; &apos;<span style="color: rgb(184, 49, 47);">student</span>&apos;</span></li>
"""
pre_process_2 = """
<ul style="list-style-type: square;">
    <li><span style="font-size: 14px;">Đối với biến c&oacute; tỉ lệ missing cao &apos;default&apos; - 20%, v&igrave; biến n&agrave;y kh&ocirc;ng c&oacute; dấu hiệu nhận biết r&otilde; r&agrave;ng n&ecirc;n nh&oacute;m quyết định kh&ocirc;ng thay đổi thuộc t&iacute;nh của biến n&agrave;y.</span></li>
</ul>
<p style="line-height: 1.5;"><span style="font-size: 14px;">4. Outlier:</span></p>
<ul style="list-style-type: square;">
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Biến age: thay thế c&aacute;c gi&aacute; trị outlier (tức lớn hơn 70) bằng 70.</span></li>
    <li style="line-height: 1.5;"><span style="font-size: 14px;">Biến campaign, previous, cons.conf.idx: thay thế những gi&aacute; trị lớn hơn quantile_95 bằng quantitle_95</span></li>
</ul>
<p style="line-height: 1.5;"><span style="font-size: 14px;">5. Ph&acirc;n loại biến:</span><span style="font-size: 14px;">Ph&acirc;n loại biến pdays th&agrave;nh 3 nh&oacute;m:&nbsp;</span></p>
<ul style="list-style-type: square; line-height: 1.5;">
    <li><span style="font-size: 14px;">&apos;not_previously_contacted&apos;: cho c&aacute;c gi&aacute; trị &ge; 999</span></li>
    <li><span style="font-size: 14px;">&apos;over_a_week&apos;: cho c&aacute;c gi&aacute; trị &ge; 7 v&agrave; &lt; 999</span></li>
    <li><span style="font-size: 14px;">&apos;within_a_week&apos;: cho c&aacute;c gi&aacute; trị &ge; 0 v&agrave; &lt; 7</span></li>
</ul>
<p><br></p>
"""
metric = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Mặc d&ugrave; nh&oacute;m sử dụng 04 c&ocirc;ng thức t&iacute;nh AC phổ biến nhất l&agrave;: Accuracy, Precision, Recall v&agrave; F1. Tuy nhi&ecirc;n, v&igrave; mục ti&ecirc;u cuối c&ugrave;ng l&agrave; l&agrave;m thế n&agrave;o để tiếp cận được nhiều kh&aacute;ch h&agrave;ng tiềm năng nhất - hay n&oacute;i c&aacute;ch kh&aacute;c l&agrave; tr&aacute;nh việc bỏ s&oacute;t kh&aacute;ch h&agrave;ng tiềm năng, nh&oacute;m quyết định sẽ quan t&acirc;m nhất đến kết quả của Recall.</span></p>
"""

overview_1 = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Mục đ&iacute;ch dự &aacute;n: Nghi&ecirc;n cứu về chiến dịch tiếp thị khuyến kh&iacute;ch kh&aacute;ch h&agrave;ng gửi tiền tiết kiệm th&ocirc;ng qua điện thoại của ng&acirc;n h&agrave;ng, từ đ&oacute; x&acirc;y dựng m&ocirc; h&igrave;nh dự đo&aacute;n về mức độ th&agrave;nh c&ocirc;ng của chiến dịch đối với nh&oacute;m kh&aacute;ch h&agrave;ng mục ti&ecirc;u v&agrave; đ&aacute;nh gi&aacute; hiệu quả kinh tế nếu &aacute;p dụng m&ocirc; h&igrave;nh đ&aacute;nh gi&aacute; v&agrave;o thực tế.</span></p>
<p style="line-height: 1.5;"><span style="font-size: 14px;">Tập Dataset: Bộ dữ liệu được d&ugrave;ng l&agrave; dữ liệu của chiến dịch tele-marketing ở Thổ Nhĩ Kỳ (từ 05/2008 đến 11/2010) tr&ecirc;n 41188 mẫu với 20 thuộc t&iacute;nh kh&aacute;c nhau.</span></p>
<p style="line-height: 1.5;"><span style="font-size: 14px;">Phương ph&aacute;p nhi&ecirc;n cứu: Đối với b&agrave;i to&aacute;n n&agrave;y, nh&oacute;m đ&atilde; thực hiện đầy đủ 7 bước/ 8 bước th&ocirc;ng thường của một dự &aacute;n data science thực tế trong doanh nghiệp như b&ecirc;n dưới:</span></p>
"""
overview_2 = """
<p style="line-height: 1.5;"><span style="font-size: 14px;">Tuy nhi&ecirc;n, v&igrave; thời gian ban tổ chức đưa ra cho dự &aacute;n c&oacute; hạn cũng như v&igrave; l&agrave; lần đầu ti&ecirc;n trải nghiệm với dự &aacute;n Data science thực tế, nh&oacute;m kh&ocirc;ng tr&aacute;nh khỏi nhiều sai s&oacute;t trong qu&aacute; tr&igrave;nh l&agrave;m. Nh&oacute;m rất mong nhận được sự th&ocirc;ng cảm, g&oacute;p &yacute; từ qu&yacute; doanh nghiệp, ban tổ chức cũng như c&aacute;c bạn tham gia chương tr&igrave;nh để nh&oacute;m c&oacute; thể học tập, sửa chữa v&agrave; ho&agrave;n thiện b&agrave;i hơn.</span></p>
"""
def main():
    
    st.title("Team 01")
    st.text(" Bài tập cuối khoá được hoàn thành bởi 04 thành viên bên dưới:")

    #Introduce team members
    col1, col2, col3, col4, col5, col6  = st.beta_columns(6)

    ava1 = Image.open("image/ava1.jpeg")
    ava2 = Image.open("image/ava2.jpeg")
    ava3 = Image.open("image/ava3.jpg")
    ava4 = Image.open("image/ava4.jpg")

    col1.header("Nguyen")
    col1.image(ava1, use_column_width=True)

    col2.header("Linh")
    col2.image(ava2, use_column_width=True)

    col3.header("Tuan")
    col3.image(ava3, use_column_width=True)

    col4.header("Hai")
    col4.image(ava4, use_column_width=True)

    # Tham khảo: https://www.google.com/search?q=write+paragraph+in+streamlit+app&oq=write+paragraph+in+streamlit+app&aqs=chrome..69i57j33i160.10955j0j4&sourceid=chrome&ie=UTF-8#kpvalbx=_Dx_eYLayJ4y9rQGQ3J6YAg53

    #Title list
    h1=  """
    <div style="background-color:#004d99;padding:0px">
    <h2 style="color:white;text-align:center;">PROJECT OVERVIEW </h2>
    </div>
    """
    h2=  """
    <div style="background-color:#004d99;padding:0px">
    <h2 style="color:white;text-align:center;">DATA OVERVIEW </h2>
    </div>
    """
    h3=  """
    <div style="background-color:#004d99;padding:0px">
    <h2 style="color:white;text-align:center;">RESULT OVERVIEW </h2>
    </div>
    """
    
    # PROJECT OVERVIEW INFORMATION
    st.markdown(h1,unsafe_allow_html=True)
    st.markdown(overview_1, True)
    col1, col2, col3 = st.beta_columns((1,3,1))
    col2.image('image/analysis_flow.png')
    st.markdown(overview_2, True)

    # DATA OVERVIEW INFORMATION
    st.markdown(h2,unsafe_allow_html=True)

    #show dataframe with filter
    marketing_w_10_rows = marketing_df[0:20]
    st.subheader("DataFrame")
    filtered = st.multiselect("Filter columns", options=list(marketing_w_10_rows.columns), default=list(marketing_w_10_rows.columns))
    st.write(marketing_w_10_rows[filtered])

    # Create selected box to display chart
    st.subheader("Data Visualization")
    ax_size = 8
    title_size = 12
    an1, an2 = st.beta_columns(2)
    info_selectbox = an1.selectbox(
    "Nội dung phân tích",
    ("Thông tin chung", "Các biến khách hàng", "Các biến phương pháp tiếp cận", "Các biến xã hội","Lưu ý chung"))

    if info_selectbox == "Thông tin chung":
        # Tỉ lệ Missing của các biến
        missing_data = missing_exploration(marketing_null)

        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,4)) 
        #Percentage of Target value
        data = marketing_df.groupby('y').size().sort_values(ascending=False)
        ax1.pie(x=data , autopct="%.1f%%", explode=[0.05]*len(data), labels=data.index.tolist());
        ax1.set_title("The percentage of Target value", fontsize=title_size);

        # Visualiza missing value percent
        sns.barplot(x=missing_data.index, y=missing_data['Percent %'], ax = ax2)
        ax2.set_title('Percent missing data by feature', fontsize=title_size)

        for ax in fig.axes:
            ax.tick_params(labelrotation=45)
        plt.tick_params(axis='both', which='major', labelsize=ax_size)
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        #Text explain
        st.markdown(general, True)

    client_varible = {'age': age,'job': job, 'marital': marital,'education': education,'default': default, 'housing': housing, 'loan': loan,"emp.var.rate": emp_var_rate, "cons.price.idx" : cons_price_idx, "cons.conf.idx" : cons_conf_idx, "euribor3m" : euribor3m, "nr.employed" : nr_employed, 'contact': contact,'day_of_week': day_of_week,'duration': duration,'campaign': campaign,'pdays': pdays,'previous': previous,'poutcome': poutcome}

    if info_selectbox == "Các biến khách hàng":
        varible_selectbox = an2.selectbox("Tên biến", ('age','job','marital','education','default', 'housing', 'loan'))
        if varible_selectbox == "age":
            visualize_numerical(marketing_df,'age',target = 'y')
            
        if varible_selectbox == "job":
            visualize_categorical_w_success(marketing_df,'job',target = 'y')

        if varible_selectbox == "marital":
            visualize_categorical_w_success(marketing_df,'marital',target = 'y')

        if varible_selectbox == "education":
            visualize_categorical_w_success_percent(marketing_df,'education',target = 'y')

        if varible_selectbox == "default":
            visualize_categorical_w_success_percent(marketing_df,'default',target = 'y')
        
        if varible_selectbox == "housing":
            visualize_categorical_w_success(marketing_df,'housing',target = 'y')
        
        if varible_selectbox == "loan":
            visualize_categorical_w_success_percent(marketing_df,'loan',target = 'y')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # visualization
        if varible_selectbox in client_varible.keys():
            st.markdown(client_varible[varible_selectbox], True)
    
    if info_selectbox == "Các biến phương pháp tiếp cận":
        bank_varible_selectbox = an2.selectbox("Tên biến",('contact','day_of_week','duration','campaign','pdays','previous','poutcome'))

        if bank_varible_selectbox == 'contact':
            visualize_categorical_w_success(marketing_df,'contact',target = 'y')

        if bank_varible_selectbox == 'day_of_week':
            visualize_categorical_w_success_percent(marketing_df,'day_of_week',target = 'y')

        if bank_varible_selectbox == 'duration':
            look_up = {'jan': 1, 'feb': 2, 'mar': 3,'apr':4,'may': 5, 'jun': 6, 'jul':7, 'aug':8,'sep': 9,'oct': 10,'nov': 11,'dec':12}
            marketing_01 = marketing_df.copy()
            marketing_01['month_num']  = marketing_01.month.map(look_up)
            #visualize
            visualize_categorical_w_success(marketing_01 , 'month_num', target = 'y')
            
        if bank_varible_selectbox == 'campaign':
            visualize_categorical_w_success(marketing_df,'campaign',target = 'y')

        if bank_varible_selectbox == 'pdays':
            data = marketing_df[marketing_df.pdays != 999]
            column = 'pdays'
            visualize_numerical(data,column)

        if bank_varible_selectbox == 'previous':
            visualize_numerical(marketing_df,'previous')

        if bank_varible_selectbox == 'poutcome':
            visualize_categorical_w_success(marketing_df,'poutcome',target = 'y')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # text explaination
        if bank_varible_selectbox in client_varible.keys():
            st.markdown(client_varible[bank_varible_selectbox], True)

    if info_selectbox == "Các biến xã hội":
        social_varible_selectbox = an2.selectbox("Tên biến", ("emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"))
        # Visualization
        visualize_numerical(marketing_df,social_varible_selectbox)
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Explain Text
        if social_varible_selectbox in client_varible.keys():
            st.markdown(client_varible[social_varible_selectbox], True)

    if info_selectbox == "Lưu ý chung":
        col1, col2  = st.beta_columns(2)
        col1.markdown(comment_1, True)
        col2.markdown(comment_2, True)

    # Add Text in explaination
    st.set_option('deprecation.showPyplotGlobalUse', False)

    ### Model result
    st.markdown(h3,unsafe_allow_html=True)        
    result_img = Image.open("image/result_visualization.png")
    result_comparison = pd.read_csv('data/AC_result.csv',sep=',')
    
    #Description about result
    st.markdown("""<p><span style="font-size: 14px;">Trong qu&aacute; tr&igrave;nh x&acirc;y dựng m&ocirc; h&igrave;nh dự đo&aacute;n v&agrave; ph&acirc;n t&iacute;ch, nh&oacute;m đ&atilde; thực hiện nhiều bước pre-process kh&aacute;c nhau dựa v&agrave;o phần ph&acirc;n t&iacute;ch data c&ugrave;ng với việc thực nghiệm tr&ecirc;n nhiều m&ocirc; h&igrave;nh kh&aacute;c nhau để dự đo&aacute;n. Tuy nhi&ecirc;n để tr&aacute;nh g&acirc;y bối rối cho người đọc, nh&oacute;m chỉ chọn lọc ra bước pre-processing được sử dụng ch&iacute;nh c&ugrave;ng 5 models cho ra kết quả cao nhất.</span></p>""", unsafe_allow_html=True)
    
    # 1. Pre-processing
    st.subheader("1. Tiền xử lí dữ liệu")
    col1, col2  = st.beta_columns(2)
    col1.markdown(pre_process_1, True)
    col2.markdown(pre_process_2, True)

    #2. Evaluation metric
    st.subheader("2. Phương pháp đánh giá")
    st.markdown(metric,True)

    # 3. result
    st.subheader("3. Kết quả")
    st.image(result_img, use_column_width=True)
    st.dataframe(result_comparison.style.highlight_max(axis=0))



    
    # if info_selectbox == "Biến liên quan đến khách hàng":


    # if info_selectbox == "Nội dung phân tích":



    # if info_selectbox == "Nội dung phân tích":
    

    df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
    
    # if add_selectbox == "Email":
    #     st.map(df)

if __name__=='__main__':
    Res=main()
    # Res=str(int(result))
    # dict={"yes":'1',"no":'0'}    
    # for i,j in dict.items():
    #     Res=Res.replace(j,i)
    
            
    # if st.sidebar.button("Show Prediction"):
    #     st.sidebar.subheader("The predicted response of customer or client to subscribe a term deposit is")
    #     st.sidebar.success(Res)
    
    


    
    
   
# streamlit run apps/data_analysis.py