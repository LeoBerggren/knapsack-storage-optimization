import pandas as pd
import numpy as np

import os
print('!!!!!!')
print(os.getcwd())
print(os.listdir('..'))
print('!!!!!!')
df = pd.read_excel('data\kidr_activity.xlsx', header=None, engine='openpyxl'
)
df = df.drop([0,1,2]) #Drops the rows that are not part of the data.
df.columns = df.iloc[0].to_list() #Designates the relevant row as the header.
df = df[1:]

var = ['SND number','Size','Days passed','#Canceled','Number of Requests', 'Access Granted*'] #relevant variables
#The size of the datasets are given in MB
df = df[var]
df = df.reset_index(drop=True)
columns_to_fill = ['Number of Requests', 'Access Granted*','#Canceled'] 
df[columns_to_fill] = df[columns_to_fill].fillna(0) #Filling in the empty rows as zeros
df['Size'] = df['Size'].astype('float')

df_ned = pd.read_excel('data/nedladdningar.xlsx', header=None, engine='openpyxl'
)
df_ned.columns = df_ned.iloc[0].to_list() #Designates the relevant row as the header.
df_ned = df_ned[1:]

var_ned = ['Dataset','Filtyp']
df_ned = df_ned[var_ned]

df_vis = pd.read_excel('data\Sidbesök_ki.xlsx', header=None, engine='openpyxl')
df_vis.columns = df_vis.iloc[0].to_list()
df_vis = df_vis[1:]
df_vis['Antal'] = df_vis['Antal'].astype(int)
Visits = 0
for v in df_vis['Antal']:
     Visits += v

#pseudocode
data = np.zeros(len(df)) #Vector of 62 zeros
doc = np.zeros(len(df))
print('datapoints: ', len(df))
Req = 0

#for each dataset (those in df) this will go through each download in df_ned for that dataset and count it as data
# or as document depending on its 'filtyp'. 
# if it is not marked as 'document' or 'data' it will raise an keyerror since something is wrong with the spreadsheet
# If it does not find anymore or any it will go to the next item. 

for index_1, value_1 in enumerate(list(df['SND number'])):
    r = df['Number of Requests'][index_1]
    Req += r
    # index_1_int: int = int(index_1)  # Explicitly cast to int
    for index_2, value_2 in enumerate(list(df_ned['Dataset']),1):
        #index_2_int: int = int(index_2)  # Explicitly cast to int
        #filtyp_value2: str = df_ned['Filtyp'][index_2_int] #explicitly declare that the value is a string.
        if value_2 == value_1:
                
                if df_ned['Filtyp'][index_2] == 'dokumentation':
                    doc[index_1] += 1
                elif df_ned['Filtyp'][index_2]== 'data':
                    data[index_1] += 1
                else:
                    print('something wrong')
        else:
            pass

data = data.astype(int)
print("data downloads: ", data)
print('Total data downloads: ', sum(data))
print('document downloads: ', doc)
print("Total document downloads: ", sum(doc))
print('total downloads:',np.sum(doc)+np.sum(data))            
print('total number of requests: ', Req)
print('total number of visits: ', Visits)



