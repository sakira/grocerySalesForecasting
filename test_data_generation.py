# test data generation
# combine all the items data from test_data_unit_sales



import pandas as pd
import numpy as np


items_meta_data = pd.read_csv('items.csv')
column_names = ['s%d' % c for c in range(1, 55)]
column_names_wos = ['%d' % c for c in range(1, 55) for i in range(0, 16)] # without store string s


for index, item in items_meta_data.iterrows():
    
    
    item_nbr = item['item_nbr']
    filename = 'test_data_unit_sales/item_' + str(item_nbr) + '.csv'
    
    
    # laod test prediction of 54 stores for item_nbr
    try:
        test_pred = pd.read_csv(filename)
        #print(test_pred.head())
        
        # rename column names
        
        # unpivot
        #test_pred_unpivot = pd.melt(test_pred, id_vars = ['A'], value_vars=['B', 'C'])
        #test_pred_unpivot = pd.melt(test_pred, id_vars = ['DS'], value_vars = ['s1', 's2'], var_name = 'store_nbr', value_name = 'yhat')
        
        # reshape
        test_pred_unpivot = pd.melt(test_pred, id_vars = ['DS'], value_vars = column_names, var_name = 'store_nbr', value_name = 'yhat')
        test_pred_unpivot.yhat[test_pred_unpivot.yhat <= 0] = 0.0
        
        
        # store in separate dataframe 
        unit_sales = pd.DataFrame({ 'date' : test_pred_unpivot['DS'],
                                    'item_nbr' : item_nbr,
                                    'store_nbr' : column_names_wos, 
                                    'unit_sales' : test_pred_unpivot['yhat'] })
        
        
             
        
        
        
        # write to a csv file with date, item_nbr, store_nbr, unit_sales
        unit_sales.to_csv('test_prediction.csv', mode = 'a', header = False, index = False)
        print('Prediction of item %d is added..' % item_nbr)
        
    except IOError: # if not exists
        print('Item %d file not found ...' % (item_nbr) )
    
    
    # counter to break the loop
    #if index > 2:
     #   break


# merge test data file and prediction files
test_data = pd.read_csv('test.csv')
pred_data = pd.read_csv('test_prediction.csv', header = None)
pred_data.columns = ['date', 'item_nbr', 'store_nbr', 'unit_sales']
print(test_data.shape)
print(pred_data.shape)
print(pred_data.head())

result = pd.merge(test_data, pred_data, how = 'left', on = ['date', 'store_nbr', 'item_nbr'])
print(result.head())
print(result.shape)

# fill the missing values
result = result.fillna(0)



# create the submission file
result.to_csv('submission_fbprophet.csv', index = False, columns = ['id', 'unit_sales'], header = True)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Ensemble of two public kernels
# For the median-based kernel, I add a multiplier of 0.95 to the original result.
# Median-based from Paulo Pinto: https://www.kaggle.com/paulorzp/log-ma-and-days-of-week-means-lb-0-529
# LGBM from Ceshine Lee: https://www.kaggle.com/ceshine/lgbm-starter

filelist = ['knn.csv', 'submission_fbprophet.csv']

outs = [pd.read_csv(f, index_col=0) for f in filelist]
concat_df = pd.concat(outs, axis=1)
concat_df.columns = ['submission1', 'submission2']
concat_df["unit_sales"] = concat_df.mean(axis = 1)
concat_df[["unit_sales"]].to_csv("ensemble.csv")





