import datetime

import pandas as pd
import numpy as np

def str_to_datetime(date_str):
    # convert DD/MM/YYYY to type datetime
    format_str = '%d/%m/%Y' # The format
    datetime_obj = datetime.datetime.strptime(date_str, format_str)
    
    return datetime_obj.date()

def get_latest_data(filename='/Users/samueltorres/Documents/Projects/Multilayer_COVID19/data/OSB_EnfTransm-COVID-19.csv'):

    cases_data = pd.read_csv(filename,encoding= 'unicode_escape', delimiter=';')
    df_cases_data = pd.DataFrame(columns=['reported_date','type'])
    reported_dates = cases_data['FECHA_DIAGNOSTICO']
    reported_dates_conv = []
    for data_date in reported_dates:
        if isinstance(data_date, str):
            date_converted = str_to_datetime(data_date)
        else: 
            date_converted = datetime.date(0000,0,0)
        reported_dates_conv.append(date_converted)

    df_cases_data['reported_date'] = reported_dates_conv
    df_cases_data['type'] = cases_data['ESTADO']

    df_latest_cases = pd.DataFrame(columns=['reported_date','type'])
    consider_from = datetime.date(2021,1,1)
    latest_dates_l = []
    latest_cases_l = []
    for idx, data_ in df_cases_data.iterrows():
        if data_['reported_date'] < consider_from:
            continue
        else:
            latest_dates_l.append(data_['reported_date'])
            latest_cases_l.append(data_['type'])

    df_latest_cases['reported_date'] = latest_dates_l
    df_latest_cases['type'] = latest_cases_l

    return df_latest_cases

def get_complete_data(filename='/Users/samueltorres/Documents/Projects/Multilayer_COVID19/data/OSB_EnfTransm-COVID-19.csv'):
    # read data
    cases_data = pd.read_csv(filename,encoding= 'unicode_escape', delimiter=';')
    # create DataFrame
    df_clean_data = pd.DataFrame(columns=['reported_date','type'])
    # extract dates
    reported_dates = cases_data['FECHA_DIAGNOSTICO']
    reported_dates_conv = []
    for data_date in reported_dates:
        if isinstance(data_date, str):
            date_converted = str_to_datetime(data_date)
        else: 
            date_converted = datetime.date(0000,0,0)
        reported_dates_conv.append(date_converted)
    # extract states
    reported_types = cases_data['ESTADO']
    reported_type_conv = []
    for data_type in reported_types:
        if data_type == 'Leve':
            reported_type_conv.append(2)
        elif data_type == 'Moderado':
            reported_type_conv.append(3)
        elif data_type == 'Grave':
            reported_type_conv.append(4)
        elif data_type == 'Fallecido':
            reported_type_conv.append(5)
        elif data_type == 'Recuperado':
            reported_type_conv.append(6)
        elif data_type == 'Fallecido (No aplica No causa Directa)':
            reported_type_conv.append(98)
        else:
            reported_type_conv.append(99)
    # save cleaned data
    df_clean_data['reported_date'] = reported_dates_conv
    df_clean_data['type'] = reported_type_conv
    # sort dates
    df_clean_data = df_clean_data.sort_values(by='reported_date')
    # create states DataFrame
    df_states_list = []
    start_date = min(df_clean_data['reported_date'])
    end_date   = max(df_clean_data['reported_date'])
    delta_d    = datetime.timedelta(days=1)
    # iterate over days
    while start_date <= end_date:
        start_date += delta_d
        actual_date = start_date
        data_i_mask = df_clean_data['reported_date'] == actual_date
        data_i      = pd.DataFrame(df_clean_data[data_i_mask])
        
        df_states_data = pd.DataFrame(columns=['reported_date','I1','I2','I3','D','R'])
        df_states_data['reported_date'] = actual_date 
        df_states_data['I1']            = sum(data_i['type'] == 2)
        df_states_data['I2']            = sum(data_i['type'] == 3)
        df_states_data['I3']            = sum(data_i['type'] == 4)
        df_states_data['D']             = sum(data_i['type'] == 5)
        df_states_data['R']             = sum(data_i['type'] == 6)

        df_states_list.append(df_states_data)

    df_return = pd.concat(df_states_list)

    return df_return
    

data = get_latest_data()
data.to_csv('data/LatestCases_Bogota.csv',index=False)

filename='/Users/samueltorres/Documents/Projects/Multilayer_COVID19/data/OSB_EnfTransm-COVID-19.csv'
cases_data = pd.read_csv(filename,encoding= 'unicode_escape', delimiter=';')
# create DataFrame
df_clean_data = pd.DataFrame(columns=['reported_date','type'])
# extract dates
reported_dates = cases_data['FECHA_DIAGNOSTICO']
reported_dates_conv = []
for data_date in reported_dates:
    if isinstance(data_date, str):
        date_converted = str_to_datetime(data_date)
    else: 
        date_converted = datetime.date(0000,0,0)
    reported_dates_conv.append(date_converted)
# extract states
reported_types = cases_data['ESTADO']
reported_type_conv = []
for data_type in reported_types:
    if data_type == 'Leve':
        reported_type_conv.append(2)
    elif data_type == 'Moderado':
        reported_type_conv.append(3)
    elif data_type == 'Grave':
        reported_type_conv.append(4)
    elif data_type == 'Fallecido':
        reported_type_conv.append(5)
    elif data_type == 'Recuperado':
        reported_type_conv.append(6)
    elif data_type == 'Fallecido (No aplica No causa Directa)':
        reported_type_conv.append(98)
    else:
        reported_type_conv.append(99)
# save cleaned data
df_clean_data['reported_date'] = reported_dates_conv
df_clean_data['type'] = reported_type_conv
# sort dates
df_clean_data = df_clean_data.sort_values(by='reported_date')
# create states DataFrame
df_states_list = []
start_date = min(df_clean_data['reported_date'])
end_date   = max(df_clean_data['reported_date'])
delta_d    = datetime.timedelta(days=1)
# iterate over days
while start_date <= end_date:
    actual_date = start_date
    data_i_mask = df_clean_data['reported_date'] == actual_date
    data_i      = pd.DataFrame(df_clean_data[data_i_mask])
    
    df_states_data = pd.DataFrame(columns=['reported_date','I1','I2','I3','D','R'])
    print(sum(data_i['type'] == 2))
    df_states_data['reported_date'] = actual_date 
    df_states_data['I1']            = sum(data_i['type'] == 2)
    df_states_data['I2']            = sum(data_i['type'] == 3)
    df_states_data['I3']            = sum(data_i['type'] == 4)
    df_states_data['D']             = sum(data_i['type'] == 5)
    df_states_data['R']             = sum(data_i['type'] == 6)

    df_states_list.append(df_states_data)

    start_date += delta_d

df_return = pd.concat(df_states_list)