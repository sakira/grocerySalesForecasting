# load libraries
import pandas as pd
import numpy as np
from fbprophet import Prophet

import sys

item_index = int(sys.argv[1])


# Add holiday effects
df_holidays_events = pd.read_csv('holidays_events.csv')
holidays = df_holidays_events[df_holidays_events['transferred'] == False][['description', 'date']]
holidays.columns = ['holiday', 'ds']
#print(holidays)



# model definition
def modelfbProphet(df_train, holidays):
    # train
    m = Prophet( changepoint_prior_scale = 2.5, yearly_seasonality = True, weekly_seasonality = True, holidays = holidays)
    m.fit(df_train)
    
    # prepare the future values to forecast
    future = m.make_future_dataframe(periods = 16)
    
    
    # prediction
    forecast = m.predict(future)
    
    return m, forecast, future
    

# load items info
items_meta_data = pd.read_csv('items.csv')
item = items_meta_data['item_nbr'][item_index]

# create the date range
DS = pd.date_range('20170101', periods = 227) #2017 Jan 01 to 2017 Aug 15

# create column names for store with date series
column_names = ['DS'] + ['s%d' % c for c in range(1, 55)]

#for index, item in items_meta_data.iterrows():
if item:    
    #if index < 9:
    #    continue
    #if index > 10:
    
    #item_nbr = item['item_nbr']
    item_nbr = item
    # load train data for each item
    #item_nbr = '105575'
    
    
    # create test dataframe for each item with shape of 54 stores and last 16 days of Aug 2017
    df_test = pd.DataFrame(columns = column_names)
    df_test['DS'] = pd.date_range('20170816', periods = 16)
    
    
    try: # if item file exist
        df = pd.read_csv('train_data_unit_sales/items_' + str(item_nbr) + '.csv')
        
        if df.empty == False:
            count = 0
            for store_i in range(1, 55):
                # create train data for each item at particular store
                store_name = 's' + str(store_i)
                df_train = pd.DataFrame({"ds": DS, "y": df[store_name]})
            
                #print(df_train.head())
            
                # fit model
                model, forecast, future = modelfbProphet(df_train, holidays)
            
                
                # save prediction data for each store
                predictions = np.round(forecast[['yhat']].tail(16))
                df_test[store_name] = predictions.values
            
                # counter for stores
                #if(count > 0):
                    #    break
                count += 1
            
            #print(df_test)
            print('Prediction for item %d is done...' % item_nbr)
    
    except IOError:
        print('Item %d doesnt exists...' % item_nbr)
    
    df_test.to_csv('test_data_unit_sales/item_' + str(item_nbr) + '.csv')
    
    if (item_index % 100) == 0:
        print('% items have been processed..' % item_index)

print('Prediction done..')

