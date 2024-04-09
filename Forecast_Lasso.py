# %% https://towardsdatascience.com/mastering-the-art-of-pricing-optimization-a-data-science-solution-eb8befb79425
# %%  DF Datas 

#df_pricing[df_pricing.isnull().any(axis=1)]
# product_num = df_modelo[cols_price_clubbi].iloc[:,0:1].columns.to_list()[0].find('-'  , 0)
# product_num = df_modelo[cols_price_clubbi].iloc[:,0:1].columns.to_list()[0].find('-'  , product_num+1) 
# produto = df_modelo[cols_price_clubbi].iloc[:,0:1].columns.to_list()[0][product_num+2:]

# df_datas = df.copy()
# df_datas = df_datas.set_index('Data')
# df_datas = df_datas[['Gmv']].groupby(df_datas.index).sum()
# #df_datas.groupby(df_datas['Data']).sum()
# #pd.dataframe(df_datas, columns = 'Data')
# df_datas = df_datas.reset_index('Data')[['Data']].set_index('Data')
# df_datas

# %% DF Erro Modelo

#df_erro_modelo = pd.read_excel('C:/Users/leona/Área de Trabalho/Prophet/Clientes/Erro_Modelo.xlsx') 
#df_erro_modelo['Data'] = df_erro_modelo['Data'].dt.date     
#df_erro_modelo = df_erro_modelo.iloc[:,:4]
#df_erro_modelo = df_erro_modelo.set_index('Data')
#df_erro_modelo = df_erro_modelo.sort_index(ascending= True)  
#df_erro_modelo = df_erro_modelo.iloc[:df_erro_modelo.shape[0]-1,:]
#df_erro_modelo

#import os

#current_directory = os.getcwd()
#current_directory


# %% Import 
import seaborn as sns
import datetime
import statsmodels.api as sm
import redshift_connector
import json
from math import sqrt 
import pandas as pd
import numpy as np  
from sklearn.linear_model import Lasso  
from feature_engine.creation import CyclicalFeatures 
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures  
from feature_engine.timeseries.forecasting import (LagFeatures,WindowFeatures,)
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tools.eval_measures import rmse
import prophet
from prophet.plot import plot_plotly, plot_components_plotly 

import matplotlib.pyplot as plt 
 
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from scipy.interpolate import interp1d

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


import mysql.connector as connection

import timeit

# %% Requerements 
# absl-py                           2.0.0
# altair                            5.2.0
# arviz                             0.11.2
# asn1crypto                        1.5.1
# astroid                           2.15.8
# asttokens                         2.4.1
# astunparse                        1.6.3
# attrs                             23.2.0
# backcall                          0.2.0
# beautifulsoup4                    4.12.3
# blinker                           1.7.0
# boto3                             1.34.65
# botocore                          1.34.65
# Bottleneck                        1.3.7
# cachetools                        5.3.3
# certifi                           2024.2.2
# cftime                            1.6.2
# charset-normalizer                3.3.2
# click                             8.1.7
# cmdstanpy                         1.2.1
# colorama                          0.4.6
# comm                              0.2.1
# convertdate                       2.4.0
# cycler                            0.12.1
# Cython                            3.0.8
# debugpy                           1.6.7
# decorator                         5.1.1
# dill                              0.3.8
# dm-tree                           0.1.8
# ephem                             4.1.2
# et-xmlfile                        1.1.0
# executing                         2.0.1
# fastjsonschema                    2.19.1
# feature-engine                    1.6.2
# flatbuffers                       24.3.7
# fonttools                         4.25.0
# gast                              0.5.4
# gitdb                             4.0.11
# GitPython                         3.1.42
# google-pasta                      0.2.0
# grpcio                            1.62.1
# h5py                              3.10.0
# holidays                          0.42
# idna                              3.6
# importlib-metadata                7.0.1
# importlib-resources               6.1.1
# ipykernel                         6.29.2
# ipython                           8.12.0
# jedi                              0.19.1
# Jinja2                            3.1.3
# jmespath                          1.0.1
# joblib                            1.3.2
# jsonschema                        4.21.1
# jsonschema-specifications         2023.12.1
# jupyter_client                    8.6.0
# jupyter_core                      4.12.0
# keras                             3.0.5
# kiwisolver                        1.4.4
# lazy-object-proxy                 1.10.0
# libclang                          16.0.6
# llvmlite                          0.42.0
# LunarCalendar                     0.0.9
# lxml                              5.1.0
# Markdown                          3.5.2
# markdown-it-py                    3.0.0
# MarkupSafe                        2.1.5
# matplotlib                        3.5.3
# matplotlib-inline                 0.1.6
# mdurl                             0.1.2
# mizani                            0.9.3
# mkl-service                       2.4.0
# ml-dtypes                         0.3.2
# munkres                           1.1.4
# mysql-connector                   2.2.9
# namex                             0.0.7
# nbformat                          5.9.2
# nest_asyncio                      1.6.0
# netCDF4                           1.6.2
# numexpr                           2.8.4
# numpy                             1.22.4
# openpyxl                          3.1.2
# opt-einsum                        3.3.0
# ortools                           9.8.3296
# packaging                         23.2
# pandas                            2.1.4
# parso                             0.8.3
# patsy                             0.5.6
# pickleshare                       0.7.5
# pillow                            10.2.0
# pip                               23.3.1
# plotly                            5.19.0
# plotnine                          0.12.4
# ply                               3.11
# prompt-toolkit                    3.0.42
# prophet                           1.1.4
# protobuf                          4.25.2
# psutil                            5.9.0
# pure-eval                         0.2.2
# pyarrow                           15.0.1
# pydeck                            0.8.1b0
# Pygments                          2.17.2
# PyMeeus                           0.5.12
# pyparsing                         3.1.1
# PyQt5                             5.15.10
# PyQt5-sip                         12.13.0
# pystan                            2.19.1.1
# python-dateutil                   2.8.2
# pytz                              2024.1
# pywin32                           227
# pyzmq                             25.1.2

# referencing                       0.33.0
# requests                          2.31.0
# requirements-detector             1.2.2
# rich                              13.7.1
# rpds-py                           0.18.0
# s3transfer                        0.10.1
# scikit-learn                      1.4.1.post1
# scipy                             1.7.3
# scramp                            1.4.4
# seaborn                           0.13.2
# semver                            3.0.2
# setuptools                        68.2.2
# sip                               6.7.12
# six                               1.16.0
# sklearn-linear-model-modification 0.0.11
# smmap                             5.0.1
# soupsieve                         2.5
# stack-data                        0.6.2
# stanio                            0.3.0
# statsmodels                       0.14.1
# stochastic                        0.7.0
# streamlit                         1.32.0
# tenacity                          8.2.3
# tensorboard                       2.16.2
# tensorboard-data-server           0.7.2
# tensorflow                        2.16.1
# tensorflow-intel                  2.16.1
# tensorflow-io-gcs-filesystem      0.31.0
# termcolor                         2.4.0
# threadpoolctl                     3.3.0
# toml                              0.10.2
# tomli                             2.0.1
# toolz                             0.12.1
# tornado                           6.2
# tqdm                              4.66.2
# traitlets                         5.14.1
# typing_extensions                 4.9.0
# tzdata                            2023.4
# urllib3                           1.26.18
# watchdog                          4.0.0
# wcwidth                           0.2.13
# Werkzeug                          3.0.1
# wheel                             0.41.2
# wrapt                             1.16.0
# xarray                            2023.2.0
# xgboost                           2.0.3
# zipp                              3.17.0

 


 
#comment many line ctrl + k + c
#remove many line ctrl + k + u



# Corte Data  =  trafego_arquivo[trafego_arquivo['Datas'] <  pd.Timestamp('2024-03-15')]   

lista_barra = ['Pedidos_window_35D_mean Scaled','Pedidos_window_21D_mean Scaled'] + ['Gmv_window_5D_mean Scaled',
       'Gmv_window_28D_mean Scaled', 'Pedidos_window_28D_mean Scaled',
       'Gmv_window_35D_mean Scaled', 'month Scaled', 'Pedidos_lag_2D Scaled', 'Pedidos_window_5D_mean Scaled',
       'Gmv_lag_35D Scaled']
  
# Funções 

def feature_dummies(df,lista,col_multiply):
  col_dummies = []
  for coluna in lista: 
    df_dummies = pd.get_dummies(df[coluna])   
    df_dummies.columns=["Gmv " + coluna + " " + str(df_dummies.columns[k-1])  for k in range(1, df_dummies.shape[1] + 1)]
    col_dummies =   col_dummies + df_dummies.columns.tolist()
    df_ = df_dummies.multiply(df[col_multiply], axis=0)
    df = df.merge(  df_dummies, how='left', left_index=True, right_index=True)   
    
  return df, col_dummies
  
#lista_dummies = ['Estado', 'Evento','Region Name','Region_id']  
#col_multiply = 'Gmv'
#coluna = 'Estado'
#df, col_dummies = feature_dummies(df,lista_dummies,col_multiply)
#df   
 


# %% Df_Eventos 
df_eventos = pd.read_excel('C:/Users/leona/Área de Trabalho/Growth/Forecast/Eventos.xlsx')
df_eventos= pd.get_dummies(df_eventos, columns=['Evento'], drop_first = False )  
df_eventos['Eventos'] = 1
colnames_event = df_eventos.columns[df_eventos.columns.str.startswith('Evento')].tolist()
df_eventos[colnames_event] = df_eventos[colnames_event].astype(float) 
df_eventos = df_eventos.set_index('Data')[['Eventos','Evento_Carnaval']]
df_eventos = df_eventos.groupby(df_eventos.index).max()
colnames_event = ['Eventos']
df_eventos  





# %% Df_Target
df_target = pd.read_excel('C:/Users/leona/Área de Trabalho/Growth/Forecast/Target_Dia.xlsx')
#df_target_dia = df_target_dia[df_target_dia['Data']>='2024-01-01']
df_target = df_target.set_index('Data')
df_target.tail(40)

# %% DF Store

df_store_id = pd.read_excel('C:/Users/leona/Área de Trabalho/Growth/Forecast/Forecast_Store_Id.xlsx')
#df = df.rename(columns={'Data': 'DataHour'  })   
#df['Data'] = df['DataHour'].dt.date     
#df['Hora'] = df['DataHour'].dt.hour      
df_store_id = df_store_id.replace(np.nan, 0 )
df_store_id = df_store_id.set_index('Data')

col_store_id = df_store_id.columns.tolist()
col_store_id

# %% Query Df 
# Query Df
 
query_order  = "select \
DATE_FORMAT(ord.order_datetime,'%Y-%m-%d %H:00:00') as DateHour, \
Date(ord.order_datetime) as Data, \
HOUR(ord.order_datetime) as Hora, \
ord.customer_id,  \
ord.region_id,  \
CASE WHEN cli.region_id in (1,7,19,27,28,29,30,31,36,37)  THEN 'RJC' \
WHEN cli.region_id in (22,24,25) THEN 'RJI' \
WHEN cli.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE ord.region_id END as 'Região',\
CONVERT(ord_ite.unit_product_id, CHAR) as Ean,\
prod.description as Produto,\
ord_ite.category as Categoria,\
concat(ord.customer_id, date_format(ord.order_datetime, 'YYYY-MM-DD')) as Pedidos,\
concat (concat(ord.customer_id, date_format(ord.order_datetime, 'YYYY-MM-DD')), ord_ite.category ) as Positivacao,\
Sum(case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end) as Quantity,\
AVG(case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end) as Price, \
SUM(ord_ite.quantity *  \
(CASE WHEN prod.gross_weight_in_gram IS NOT NULL THEN prod.gross_weight_in_gram  WHEN prod.net_volume_in_liters IS NOT NULL AND cat.gross_weight_per_content_volume_liter IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_volume_in_liters * cat.gross_weight_per_content_volume_liter  \
WHEN prod.net_weight_in_gram IS NOT NULL AND cat.gross_weight_per_net_weight_gram IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_weight_in_gram * cat.gross_weight_per_net_weight_gram  \
WHEN cat.category_id IS NOT NULL THEN COALESCE(cat.average_without_outliers, cat.average) * COALESCE(prod.number_of_items, 1)  \
WHEN prod.net_weight_in_gram IS NOT NULL THEN prod.net_weight_in_gram  \
WHEN prod.net_volume_in_liters IS NOT NULL THEN prod.net_volume_in_liters * 1000 ELSE 1000  END)) / 1000.0 as 'Peso',  \
SUM(  ord_ite.total_price  ) 'Gmv' \
from  clubbi_backend.order ord \
left join clubbi_backend.order_item ord_ite on ord_ite.order_id = ord.id and (ord_ite.is_cancelled = 0 or ord_ite.is_cancelled IS NULL) \
left join clubbi.product prod ON ord_ite.product_id = prod.ean \
left join  clubbi.merchants  cli on cli.client_site_code = ord.customer_id \
left join clubbi.category_volume cat ON prod.category_id = cat.category_id   \
where    \
1 = 1 \
and DATE(ord.order_datetime) >= '2024-03-01'  \
and DATE(ord.order_datetime) < '2025-01-01'  \
group by 1,2,3,4,5,6,7,8,9;"
 
query_order  = "select \
DATE_FORMAT(ord.order_datetime,'%Y-%m-%d %H:00:00') as DateHour, \
Date(ord.order_datetime) as Data, \
HOUR(ord.order_datetime) as Hora, \
ord.id as order_id,\
ord_ite.id as order_item_id, \
ord.customer_id, \
ord.region_id, \
CASE WHEN cli.region_id in (1,7,19,27,28,29,30,31,36,37)  THEN 'RJC' \
WHEN cli.region_id in (22,24,25) THEN 'RJI' \
WHEN cli.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE ord.region_id END as 'Região',\
CONVERT(ord_ite.unit_product_id, CHAR) as Ean,\
prod.description as Produto,\
ord_ite.category as Categoria,\
concat(ord.customer_id, date_format(ord.order_datetime, 'YYYY-MM-DD')) as Pedidos,\
concat (concat(ord.customer_id, date_format(ord.order_datetime, 'YYYY-MM-DD')), ord_ite.category ) as Positivacao,\
Sum(case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end) as Quantity,\
AVG(case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end) as Price, \
SUM(ord_ite.quantity *  \
(CASE WHEN prod.gross_weight_in_gram IS NOT NULL THEN prod.gross_weight_in_gram  WHEN prod.net_volume_in_liters IS NOT NULL AND cat.gross_weight_per_content_volume_liter IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_volume_in_liters * cat.gross_weight_per_content_volume_liter  \
WHEN prod.net_weight_in_gram IS NOT NULL AND cat.gross_weight_per_net_weight_gram IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_weight_in_gram * cat.gross_weight_per_net_weight_gram  \
WHEN cat.category_id IS NOT NULL THEN COALESCE(cat.average_without_outliers, cat.average) * COALESCE(prod.number_of_items, 1)  \
WHEN prod.net_weight_in_gram IS NOT NULL THEN prod.net_weight_in_gram  \
WHEN prod.net_volume_in_liters IS NOT NULL THEN prod.net_volume_in_liters * 1000 ELSE 1000  END)) / 1000.0 as 'Peso',  \
SUM(  ord_ite.total_price  ) 'Gmv' \
from  clubbi_backend.order ord \
left join clubbi_backend.order_item ord_ite on ord_ite.order_id = ord.id and (ord_ite.is_cancelled = 0 or ord_ite.is_cancelled IS NULL) \
left join clubbi.product prod ON ord_ite.product_id = prod.ean \
left join  clubbi.merchants  cli on cli.client_site_code = ord.customer_id \
left join clubbi.category_volume cat ON prod.category_id = cat.category_id   \
where    \
1 = 1 \
and DATE(ord.order_datetime) >= '2024-03-01'  \
and DATE(ord.order_datetime) < '2025-01-01'  \
;"

 
query_order  = "select \
DATE_FORMAT(ord.order_datetime,'%Y-%m-%d %H:00:00') as DateHour,\
Date(ord.order_datetime) as Data,\
HOUR(ord.order_datetime) as Hora,\
CONVERT(ord.id, char) as order_id,\
CONVERT(ord_ite.id, char) as order_item_id,\
ord.customer_id, \
ord.region_id, \
CASE WHEN cli.region_id in (1,7,19,27,28,29,30,31,36,37)  THEN 'RJC' \
WHEN cli.region_id in (22,24,25) THEN 'RJI' \
WHEN cli.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE ord.region_id END as 'Região',\
CONVERT(ord_ite.product_id, CHAR) as ean,\
CONVERT(ord_ite.unit_product_id, CHAR) as unit_ean,\
prod.description as Produto,\
ord_ite.category as Categoria,\
discounts.reason as discount_reason,\
discounts.operator as discount_operator,\
case when discounts.reason is not null then 1 else 0 end as flag_desconto,\
ord_ite.is_multi_package,\
ord_ite.product_package_qtd,\
ord_ite.price_managers,\
Convert(ord_ite.offer_id, char) as offer_id, \
offer.source as offer_source,\
offer.is_ofertao,\
offer.limit_quantity,\
offer.limit_per_order,\
offer.is_priority,\
case when  ord_ite.is_multi_package = 0 then  COALESCE(offer.price,discounts.original_price,ord_ite.unit_price) else COALESCE(offer.price,discounts.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd  end as offer_price, \
case when  ord_ite.is_multi_package = 0 then  COALESCE(discounts.original_price,ord_ite.unit_price) else COALESCE(discounts.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd  end as Original_Price, \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end as Price, \
case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end as Quantity, \
ord_ite.quantity *  \
(CASE WHEN prod.gross_weight_in_gram IS NOT NULL THEN prod.gross_weight_in_gram  WHEN prod.net_volume_in_liters IS NOT NULL AND cat.gross_weight_per_content_volume_liter IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_volume_in_liters * cat.gross_weight_per_content_volume_liter  \
WHEN prod.net_weight_in_gram IS NOT NULL AND cat.gross_weight_per_net_weight_gram IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_weight_in_gram * cat.gross_weight_per_net_weight_gram  \
WHEN cat.category_id IS NOT NULL THEN COALESCE(cat.average_without_outliers, cat.average) * COALESCE(prod.number_of_items, 1)  \
WHEN prod.net_weight_in_gram IS NOT NULL THEN prod.net_weight_in_gram  \
WHEN prod.net_volume_in_liters IS NOT NULL THEN prod.net_volume_in_liters * 1000 ELSE 1000  END) / 1000.0 as 'Peso',\
ord_ite.total_price  as 'Gmv' \
from  clubbi_backend.order ord \
left join clubbi_backend.order_item ord_ite on ord_ite.order_id = ord.id and (ord_ite.is_cancelled = 0 or ord_ite.is_cancelled IS NULL) \
left join clubbi.product prod ON ord_ite.product_id = prod.ean \
left join  clubbi.merchants  cli on cli.client_site_code = ord.customer_id \
left join clubbi.category_volume cat ON prod.category_id = cat.category_id  \
left join clubbi_backend.discounted_prices discounts on CONVERT(discounts.order_item_id, char) = CONVERT(ord_ite.id , char) \
left join clubbi.offer offer ON Convert(offer.id, char) = Convert(ord_ite.offer_id, char) \
where    \
1 = 1 \
and DATE(ord.order_datetime) >= '2019-01-01' \
and DATE(ord.order_datetime) < '2022-01-01'  \
case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd  end as Original_Price, \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end as Price, \
case when ord_ite.original_price > ord_ite.unit_price then\
case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end -\
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end \
else 0 end as desconto_unitario, \
case when ord_ite.original_price > ord_ite.unit_price then \
(case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end ) * \
case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end \
else 0 end as desconto_total, \
;"\

 
query_order  = "select \
DATE_FORMAT(ord.order_datetime,'%Y-%m-%d %H:00:00') as DateHour,\
Date(ord.order_datetime) as Data,\
HOUR(ord.order_datetime) as Hora,\
CONVERT(ord.id, char) as order_id,\
CONVERT(ord_ite.id, char) as order_item_id,\
ord.customer_id, \
ord.region_id, \
CASE WHEN cli.region_id in (1,7,19,27,28,29,30,31,36,37)  THEN 'RJC' \
WHEN cli.region_id in (22,24,25) THEN 'RJI' \
WHEN cli.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE ord.region_id END as 'Região',\
ord_ite.store_id, \
CONVERT(ord_ite.product_id, CHAR) as ean,\
CONVERT(ord_ite.unit_product_id, CHAR) as unit_ean,\
prod.description as Produto,\
ord_ite.category as Categoria,\
ord_ite.is_multi_package,\
ord_ite.product_package_qtd,\
ord_ite.price_managers,\
Convert(ord_ite.offer_id, char) as offer_id,  \
case when ord_ite.original_price > ord_ite.unit_price then 1 else 0 end as flag_desconto,\
case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd  end as Original_Price, \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end as Price, \
case when ord_ite.original_price > ord_ite.unit_price then \
case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end \
else 0 end as desconto_unitario, \
case when ord_ite.original_price > ord_ite.unit_price then \
(case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end ) * \
case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end \
else 0 end as desconto_total, \
case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end as Quantity, \
ord_ite.quantity *  \
(CASE WHEN prod.gross_weight_in_gram IS NOT NULL THEN prod.gross_weight_in_gram  WHEN prod.net_volume_in_liters IS NOT NULL AND cat.gross_weight_per_content_volume_liter IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_volume_in_liters * cat.gross_weight_per_content_volume_liter  \
WHEN prod.net_weight_in_gram IS NOT NULL AND cat.gross_weight_per_net_weight_gram IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_weight_in_gram * cat.gross_weight_per_net_weight_gram  \
WHEN cat.category_id IS NOT NULL THEN COALESCE(cat.average_without_outliers, cat.average) * COALESCE(prod.number_of_items, 1)  \
WHEN prod.net_weight_in_gram IS NOT NULL THEN prod.net_weight_in_gram  \
WHEN prod.net_volume_in_liters IS NOT NULL THEN prod.net_volume_in_liters * 1000 ELSE 1000  END) / 1000.0 as 'Peso',\
ord_ite.total_price  as 'Gmv' \
from  clubbi_backend.order ord \
left join clubbi_backend.order_item ord_ite on ord_ite.order_id = ord.id and (ord_ite.is_cancelled = 0 or ord_ite.is_cancelled IS NULL) \
left join clubbi.product prod ON ord_ite.product_id = prod.ean \
left join  clubbi.merchants  cli on cli.client_site_code = ord.customer_id \
left join clubbi.category_volume cat ON prod.category_id = cat.category_id  \
where    \
1 = 1 \
and DATE(ord.order_datetime) >= '2024-03-01' \
and DATE(ord.order_datetime) < '2025-03-01'  \
;"\


query_produtos  = "select convert(prod.ean,char) as ean ,prod.description,prod.category_id, prod.unit_ean, prod.only_sell_package, cat.category as Categoria, cat.section  from clubbi.product prod left join clubbi.category cat on cat.id = prod.category_id ;"


mydb =  connection.connect(
    host="aurora-mysql-db.cluster-ro-cjcocankcwqi.us-east-1.rds.amazonaws.com",
    user="ops-excellence-ro",
    password="L5!jj@Jm#9J+9K"
)

query_produtos = pd.read_sql(query_produtos,mydb) 
query_orders = pd.read_sql(query_order,mydb) 

mydb.close() #close the connection

    
df2022 = pd.read_csv('C:/Users/leona/Área de Trabalho/Prophet/Vendas_2022.csv' ) 
df202301 = pd.read_csv('C:/Users/leona/Área de Trabalho/Prophet/Vendas_2023_01.csv' ) 
df202302 = pd.read_csv('C:/Users/leona/Área de Trabalho/Prophet/Vendas_2023_02.csv' ) 
df2024Fev = pd.read_csv('C:/Users/leona/Área de Trabalho/Prophet/Vendas_2024_Fev.csv' ) 
df_passado = pd.concat([df2022,df202301,df202302,df2024Fev])
df_inicial = pd.concat([df_passado.iloc[:,1:],query_orders])


df_inicial['Quantity'] = df_inicial['Quantity'].replace(np.nan,0)
df_inicial = df_inicial[df_inicial['Quantity'] > 0]

df_inicial['ean'] = df_inicial['ean'].astype(np.int64).astype(str) 
df_inicial['unit_ean'] = df_inicial['unit_ean'].astype(np.int64).astype(str) 
 

df_inicial = df_inicial.drop(columns = ['Categoria'])

 
df_produtos = query_produtos.copy()
df_produtos['ean'] = df_produtos['ean'].astype(np.int64).astype(str)
df_produtos = df_produtos.rename(columns={'ean':'unit_ean_prod','description':'Unit_Description'})[['unit_ean_prod','Unit_Description','Categoria']]  



df_inicial = df_inicial.merge(df_produtos  ,how ='left', left_on='unit_ean', right_on='unit_ean_prod', suffixes=(False, False))
df_inicial['Categoria'] =   np.where((df_inicial['Categoria'] == 'Óleos, Azeites e Vinagres') ,  'Óleos, Azeites E Vinagres'  , df_inicial['Categoria'] )
df_inicial['price_managers'] = df_inicial['price_managers'].replace(np.nan, 0 )
df_inicial['offer_id'] = df_inicial['offer_id'].replace(np.nan, 0 ).astype(np.int64).astype(float)
 
#df_inicial['Categoria'] = np.where((df_inicial['unit_ean'] == '7896036090244') , 'Óleos, Azeites E Vinagres' , df_inicial['Categoria'] )
df_inicial

 



# %% Query Ofertão 
# Query Ofertão
 
query_ofertao  = "select \
o.date_start as Data, \
p.unit_ean as Ean, \
cat.category, \
p.description as Description, \
s.type, \
rs.region_id, \
CASE WHEN rs.region_id in (1,7,19,27,28,29,30,31,36,37)  THEN 'RJC' \
WHEN rs.region_id   in (22,24,25) THEN 'RJI' \
WHEN rs.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE rs.region_id  END as 'Região',\
o.price \
FROM clubbi.offer o \
left JOIN clubbi.product p on p.id = o.product_id \
left join clubbi.supplier s on s.id = o.supplier_id \
left join clubbi.region_supplier rs on rs.supplier_id = s.id \
left join clubbi.category cat on cat.id = p.category_id \
where \
o.is_ofertao = 1 \
and DATE(o.date_start) >= '2024-03-01' \
and DATE(o.date_start) < '2025-01-01' \
;"\

  
#Your statements here
 


mydb =  connection.connect(
    host="aurora-mysql-db.cluster-ro-cjcocankcwqi.us-east-1.rds.amazonaws.com",
    user="ops-excellence-ro",
    password="L5!jj@Jm#9J+9K"
)

df_ofertao_query = pd.read_sql(query_ofertao,mydb) 
mydb.close() #close the connection
  
#df_ofertao.to_csv('C:/Users/leona/Área de Trabalho/Prophet/Clientes/Ofertao2024.csv')  
 
df_ofertao_2024 = pd.read_csv('C:/Users/leona/Área de Trabalho/Prophet/Clientes/Ofertao2024.csv' )  
df_ofertao_inicial = pd.concat([df_ofertao_2024.iloc[:,1:],df_ofertao_query])
df_ofertao_inicial['Data'] = pd.to_datetime(df_ofertao_inicial['Data'])
df_ofertao_inicial['Data'] = df_ofertao_inicial['Data'].dt.date

 











# %% Query Users
 
#df_users_incial = pd.read_excel('C:/Users/leona/Área de Trabalho/Prophet/Clientes/Users.xlsx')  
 
conn = redshift_connector.connect(
    host='redshift-analytics-cluster-1.c8ccslr41yjs.us-east-1.redshift.amazonaws.com',
    database='dev',
    user='pbi_user',
    password='4cL6z0E7wiBpAjNRlqKkFiLW'
)
cursor: redshift_connector.Cursor = conn.cursor()
#query =  '''select *, case when region_id in (49,50,51,52,53) then 'BAC' when region_id in (1,7,19,27,28,29,30,31,36,37) then 'RJC' when region_id in (22,24,25) then 'RJI' else '-' end as "Region Name"  from public.ops_customer '''
query =  '''select * from public.ops_customer '''
cursor.execute(query)
df_users_incial: pd.DataFrame = cursor.fetch_dataframe() 
df_users_incial

# %% Query Concorrencia
# Query Concorrencia 


import redshift_connector

conn = redshift_connector.connect(
    host='redshift-analytics-cluster-1.c8ccslr41yjs.us-east-1.redshift.amazonaws.com',
    database='dev',
    user='pbi_user',
    password='4cL6z0E7wiBpAjNRlqKkFiLW'
)
cursor: redshift_connector.Cursor = conn.cursor()
query =  '''select * from public.concorrencia_python '''
cursor.execute(query)
query_concorrencia: pd.DataFrame = cursor.fetch_dataframe() 


 
#df_concorrencia = pd.read_csv('C:/Users/leona/Área de Trabalho/Prophet/Concorrencia.csv')
query_concorrencia['data'] = pd.to_datetime(query_concorrencia['data'])
query_concorrencia.sort_values('data',ascending=True )
query_concorrencia['ean'] = query_concorrencia['ean'].astype(str) 
query_concorrencia
 
# %% Query Trafego 

import redshift_connector

conn = redshift_connector.connect(
    host='redshift-analytics-cluster-1.c8ccslr41yjs.us-east-1.redshift.amazonaws.com',
    database='dev',
    user='pbi_user',
    password='4cL6z0E7wiBpAjNRlqKkFiLW'
)
cursor: redshift_connector.Cursor = conn.cursor()
query =  '''select * from public.trafego_site_hours where datas>='2024-03-15' '''
cursor.execute(query)
df_trafego_query: pd.DataFrame = cursor.fetch_dataframe() 
df_trafego_query

# %% Query Trafego Produto


# import redshift_connector

# conn = redshift_connector.connect(
#     host='redshift-analytics-cluster-1.c8ccslr41yjs.us-east-1.redshift.amazonaws.com',
#     database='dev',
#     user='pbi_user',
#     password='4cL6z0E7wiBpAjNRlqKkFiLW'
# )
# cursor: redshift_connector.Cursor = conn.cursor()
# query =  '''select * from public.trafego_site_top_produtos '''
# cursor.execute(query)
# query_trafego_produto: pd.DataFrame = cursor.fetch_dataframe() 
# query_trafego_produto



 
# %% Df Users


df_users =  df_users_incial.copy() 
#df_users = df_users[['Region Name','client_site_code']].groupby( df_users['Region Name']).agg({ 'client_site_code': pd.Series.nunique })

 
#df_users.iloc[:, 30:40].head(4) 

df_users['Tipo_Cliente'] = np.where(
   
                                      (df_users['size']== 'counter') |  
                                      (df_users['size']== 'one_checkout') |  
                                      (df_users['size']== 'two_checkouts') | 
                                      (df_users['size']== 'three_to_four_checkouts'),
                                      '1-4 Cxs' , '5-9 Cxs' )



df_users['1-4 Cxs'] = np.where((df_users['Tipo_Cliente'] == '1-4 Cxs') , 1 , 0 )
df_users['5-9 Cxs'] = np.where((df_users['Tipo_Cliente'] == '1-4 Cxs') , 0 , 1 )
df_users = df_users.rename(columns = {'region name':'Region Name'})  
df_users['Não_Mercado'] =  np.where((df_users['tipo da loja'] == 'Mercado' ) ,   0, 1  )
df_users





# %% Df Inicial 
# Dataframe Df


df = df_inicial.copy()  
df['Data'] = pd.to_datetime(df['Data'])     
df = df[df['Região']== 'RJC']  
  
 

region = 0
data_final =   pd.Timestamp(datetime.date.today()) 
 

 
lista_categorias = ['Leite']
lista_categorias = ['Óleos, Azeites E Vinagres']
lista_categorias = ['Óleos, Azeites E Vinagres','Açúcar e Adoçante','Leite','Cervejas', 'Chocolates' , 'Arroz e Feijão','Biscoitos','Derivados de Leite']

lista_categorias = ['Açúcar e Adoçante', 'Leites e Derivados','Óleos, Azeites E Vinagres','Leite', 'Arroz e Feijão','Biscoitos', 'Refrigerantes',  'Cervejas', 'Cafés, Chás e Achocolatados','Derivados de Leite']

lista_categorias = ['Óleos, Azeites E Vinagres']

lista_categorias = ['Leite'] 


# %% Df Skus  
# Df Skus  



df_top_skus = df.copy() 
  
df_top_skus = df_top_skus.iloc[int(df.shape[0] *0.5):,:]
df_top_skus = df_top_skus[df_top_skus['Categoria'].isin(lista_categorias)]
df_top_skus = df_top_skus[['Categoria','unit_ean','Unit_Description','Gmv']].groupby(['Categoria', 'unit_ean','Unit_Description']).sum().sort_values(by =['Categoria','Gmv'],ascending= False)
 
 

df_top_skus = df_top_skus.reset_index(drop=False)
df_gmv_categoria = df_top_skus[['Categoria','Gmv']].groupby('Categoria').sum().reset_index(drop = False) 
df_gmv_categoria = df_gmv_categoria.sort_values('Gmv',ascending = False) 
#df_gmv_categoria.head(10)['Categoria'].to_list()

 
df_gmv_categoria = df_gmv_categoria.rename(columns = {'Gmv':'Gmv_Categoria'})
 
 

df_top_skus  = df_top_skus.merge(df_gmv_categoria , how ='left', left_on='Categoria', right_on='Categoria', suffixes=(False, False))
df_top_skus['Share_Produto'] = df_top_skus['Gmv']/df_top_skus['Gmv_Categoria']
df_top_skus['Share_Acumulado'] = df_top_skus.groupby(['Categoria'])['Share_Produto'].cumsum()
df_top_skus['Ranking'] = 1 
df_top_skus['Ranking'] = df_top_skus.groupby(['Categoria'])['Ranking'].cumsum()
df_top_skus = df_top_skus[df_top_skus['Ranking']<=5]
df_top_skus = df_top_skus[df_top_skus['Share_Acumulado']<=0.9].sort_values('Gmv',ascending=False).reset_index(drop=True).reset_index(drop=False)
 
top_skus = df_top_skus['unit_ean'].unique().tolist()
top_skus
top_skus_description = df_top_skus['Unit_Description'].unique().tolist()

df_top_categorias = df_top_skus[['Categoria','Ranking']].groupby('Categoria').count()
df_top_categorias = df_top_categorias.rename( columns = {'Ranking':'Top Skus'})
df_top_categorias =  df_top_categorias.reset_index(drop = False)
 
df_top_categorias['key_categoria'] = 1 
df_top_categorias = df_top_categorias.set_index('key_categoria')


df_top_categorias  = pd.pivot_table(df_top_categorias , values=['Top Skus'], index=['key_categoria'], columns=['Categoria'],aggfunc={ 'Top Skus': [ "median" ]})
df_top_categorias.columns = df_top_categorias .columns.droplevel(0)
df_top_categorias.columns = df_top_categorias .columns.droplevel(0)
 


df_top_categorias.columns=[ "Top Produtos Total Categoria_" +   str(df_top_categorias.columns[k-1]) for k in range(1, df_top_categorias.shape[1] + 1)]


df_top_categorias  = df_top_categorias.reset_index(drop = False).set_index('key_categoria')
df_top_categorias  = pd.DataFrame(df_top_categorias.to_records()).set_index('key_categoria')
df_top_categorias = df_top_categorias.reset_index(drop = False)

df_top_categorias


# %%  DF Trafego


 
  
df_trafego_atual = df_trafego_query.copy()
df_trafego_atual.columns=[ str(df_trafego_atual.columns[k-1]).title()  for k in range(1, df_trafego_atual.shape[1] + 1)]
df_trafego_atual = df_trafego_atual[['Datas', 'Datetimes', 'Hora', 'User_Id', 'Chave_Cliente_Dia', 'Chave_Final','Acessos' , 'Trafego', 'Search_Products', 'Add_To_Cart','Checkout']]

df_trafego_atual = df_trafego_atual.sort_values('Datas',ascending = True)

#df_trafego_atual.to_csv('C:/Users/leona/Área de Trabalho/Prophet/Clientes/Trafego_ate_14_03.csv')  
trarego_back = pd.read_csv('C:/Users/leona/Área de Trabalho/Prophet/Clientes/Trafego_ate_14_03.csv' )  
df_trafego = pd.concat([trarego_back.iloc[:,1:],df_trafego_atual])

df_trafego = df_trafego.rename(columns = {'User_Id':'User','Datas':'Data', 'Search_Products':'Trafego_Search_Products', 'Add_To_Cart':'Trafego_Add_To_Cart', 'Checkout':'Trafego_Checkout' })

#df_trafego = df_trafego.set_index('Data')
df_trafego = df_trafego.drop(columns = ['Datetimes','Hora','Chave_Final','Chave_Cliente_Dia'])
df_trafego['Data'] = pd.to_datetime(df_trafego['Data'])
df_trafego = df_trafego.groupby(['Data','User']).sum().reset_index(drop = False) 
 

df_trafego['key'] = df_trafego['Data'].astype(str) + df_trafego['User'].astype(str)
#df_trafego = df_trafego.interpolate(limit_direction='both') 

df_trafego['Trafego'] = np.where((df_trafego['Trafego'] > 0) ,  1  , 0 )
df_trafego['Trafego_Search_Products'] = np.where((df_trafego['Trafego_Search_Products'] > 0) ,  1  , 0 )
df_trafego['Trafego_Add_To_Cart'] = np.where((df_trafego['Trafego_Add_To_Cart'] > 0) ,  1  , 0 )
df_trafego['Trafego_Checkout'] = np.where((df_trafego['Trafego_Checkout'] > 0) ,  1  , 0 )
df_trafego = df_trafego.drop(columns = ['Acessos'])
df_trafego

 





 


# %% DF Ofertão


df_ofertao = df_ofertao_inicial.copy() 
df_ofertao = df_ofertao[df_ofertao['Região'] == 'RJC']  

df_ofertao = df_ofertao[['Data','category','price']].groupby(['Data','category']).min('price')
df_ofertao = df_ofertao.reset_index(drop = False)
df_ofertao = df_ofertao.set_index('Data') 
 
df_ofertao = pd.get_dummies(df_ofertao['category']).astype(float)
df_ofertao = df_ofertao.groupby('Data').max() 
df_ofertao.columns=[ "Ofertão " + str(df_ofertao.columns[k-1]).title()  for k in range(1, df_ofertao.shape[1] + 1)]
df_ofertao['Ofertão'] = 1 

 
# var_leite = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Leite')].tolist()
# var_oleo = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Óleo')].tolist()
# var_acucar = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Açúcar E Adoçante')].tolist()


# var_biscoito = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Biscoito')].tolist()
# var_chocolate = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Chocolate')].tolist()

# var_arroz_feijao = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Arroz')].tolist()
# var_cafe = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Cafés')].tolist()



# var_cerveja = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Cervejas')].tolist()
# var_energetico = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Energéticos')].tolist()
# var_der_leite = df_ofertao.columns[df_ofertao.columns.str.startswith('Ofertão Derivados De Leite ')].tolist()

  
df_ofertao_dia = df_ofertao.copy() 
# df_ofertao = df_ofertao[['Ofertão']+  var_oleo + var_leite + var_acucar + var_der_leite + var_biscoito + var_energetico + var_cerveja +  var_chocolate + var_arroz_feijao + var_cafe ]



ofertao_columns = list(map(lambda x: 'Ofertão ' + x,lista_categorias ))
ofertao_columns  = ofertao_columns + ['Ofertão']
 
df_ofertao = df_ofertao[ofertao_columns]
df_ofertao_dia = df_ofertao_dia.iloc[df_ofertao_dia.shape[0]-10:,:]
df_ofertao_dia = df_ofertao_dia[df_ofertao_dia['Ofertão']==1].T.astype(float)
df_ofertao_dia['Total'] =   df_ofertao_dia.iloc[:, :].sum(axis=1)
df_ofertao_dia.sort_values('Total',ascending= False) 
 
df_ofertao_prod = df_ofertao_inicial.copy() 
df_ofertao_prod
 
df_ofertao_prod = df_ofertao_prod[df_ofertao_prod['Região'] == 'RJC']  

df_ofertao_prod = df_ofertao_prod[['Data','category','Ean','Description','price']].groupby(['Data','category','Ean','Description']).min('price')
df_ofertao_prod = df_ofertao_prod.reset_index(drop = False)
df_ofertao_prod = df_ofertao_prod.set_index('Data') 
df_ofertao_prod['Ean'] = df_ofertao_prod['Ean'].astype(str)
df_ofertao_prod = df_ofertao_prod[df_ofertao_prod['Ean'].isin(top_skus)]
df_ofertao_prod = df_ofertao_prod.reset_index(drop = False)
df_ofertao_prod = df_ofertao_prod.merge(df_produtos  ,how ='left', left_on='Ean', right_on='unit_ean_prod', suffixes=(False, False))

 
df_ofertao_prod = df_ofertao_prod[df_ofertao_prod['Description'] ==  df_ofertao_prod['Unit_Description']]
df_ofertao_prod = df_ofertao_prod[['Data','category','Ean','Description','price']]
df_ofertao_prod['Product_Ofertao'] = 'Ofertão ' +  df_ofertao_prod['category'].astype(str) + ' ' + df_ofertao_prod['Description'].astype(str) + ' - ' + df_ofertao_prod['Ean'].astype(str)
df_ofertao_prod = df_ofertao_prod[['Data']].merge( pd.get_dummies(df_ofertao_prod['Product_Ofertao']).astype(float)  ,how ='left', left_index= True, right_index= True, suffixes=(False, False))
df_ofertao_prod = df_ofertao_prod.groupby('Data').max()
df_ofertao_prod

df_ofertao = df_ofertao.merge(df_ofertao_prod  ,how ='left', left_index = True , right_index = True, suffixes=(False, False))
df_ofertao =  pd.DataFrame(df_ofertao.asfreq('D').index).set_index('Data').merge(df_ofertao, left_index = True, right_index=True,how = "left") 
df_ofertao = df_ofertao.replace(np.nan, 0) 

ofertao_columns = df_ofertao.columns.to_list()



df_ofertao
 

# %%  Df Vendas
df_vendas = df.copy() 
df_vendas['key'] = df_vendas['Data'].astype(str) + df_vendas['customer_id'].astype(str)
df_vendas['Pedidos'] = df_vendas['key'] 
df_vendas['Positivação'] = df_vendas['Data'].astype(str) + df_vendas['customer_id'].astype(str) + df_vendas['Categoria'].astype(str)
df_vendas = df_vendas.merge( df_users[['client_site_code','Region Name','Tipo_Cliente','5-9 Cxs','1-4 Cxs']] ,how ='left', left_on='customer_id', right_on='client_site_code', suffixes=(False, False))
df_vendas = df_vendas.drop(columns=['client_site_code'])
df_vendas['Gmv 5-9 Cxs'] = df_vendas['Gmv'].multiply(df_vendas['5-9 Cxs'] , axis=0)
df_vendas['Gmv 1-4 Cxs'] = df_vendas['Gmv'].multiply(df_vendas['1-4 Cxs'] , axis=0)
df_vendas = df_vendas.groupby( df_vendas['key']).agg({'Gmv':'sum', 'Quantity':'sum' , 'Pedidos': pd.Series.nunique , 'Gmv 5-9 Cxs':'sum','Gmv 1-4 Cxs':'sum' }).reset_index(drop= False)
df_vendas = df_vendas[['key','Gmv','Pedidos','Quantity']]
 


# %%  DF Concorrencia
# Df Concorrencia 

df_concorrencia = query_concorrencia.copy()
df_concorrencia = df_concorrencia[df_concorrencia['ean'].isin(top_skus)].drop(columns=['categoria','description','region_name'])
price_cols = df_concorrencia.drop(columns = ['data','ean']).columns.to_list()
df_concorrencia = df_concorrencia.merge(df_produtos  ,how ='left', left_on='ean', right_on='unit_ean_prod', suffixes=(False, False))
#df_concorrencia = df_concorrencia[['data','ean','Unit_Description']] 

cols_conco = df_concorrencia.drop(columns=['Unit_Description','Categoria']).columns.to_list()
df_concorrencia['Produtos'] = df_concorrencia['Unit_Description'].astype(str)  + " - " + df_concorrencia['ean'].astype(str)
df_concorrencia = df_concorrencia[['Produtos','Unit_Description'] + cols_conco]
df_concorrencia = df_concorrencia.set_index('data')
df_concorrencia  

  
price_cols = ['price_avg_concorrencia','price_guanabara','price_ceasa','price_mundial']
price_cols = ['price_mundial','price_ceasa','price_torre_cia','price_nova_coqueiro_alimentos']

#price_cols = ['price_mundial']
#df_concorrencia = df_concorrencia[['data','ean','Unit_Description'] + price_cols] 

df_concorrencia.iloc[:,4:] = df_concorrencia.iloc[:,4:].astype(float)
df_concorrencia_final = df_concorrencia.copy()  
df_concorrencia_final = df_concorrencia_final[  price_cols]

 
 
df_concorrencia_dummies = pd.get_dummies(df_concorrencia['Produtos'] )
df_concorrencia_dummies = df_concorrencia_dummies.astype(float)
df_concorrencia_dummies = df_concorrencia_dummies.replace(0,np.nan)
df_concorrencia_dummies

   
for k in range(1, df_concorrencia.shape[1] + 1): 
   
  if df_concorrencia.columns[k-1] in price_cols:
    print(df_concorrencia.columns[k-1])
    concorrente = df_concorrencia.columns[k-1][6:]
    df_concorrencia_prices = df_concorrencia_dummies.multiply(df_concorrencia['price_' + concorrente ], axis=0)     
  
    df_concorrencia_prices.columns=[concorrente.title() + " " + str(df_concorrencia_prices.columns[k-1])  for k in range(1, df_concorrencia_prices.shape[1] + 1)]
    df_concorrencia_final =  df_concorrencia_final.merge(df_concorrencia_prices, left_index = True, right_index=True,how = "left") 
    df_concorrencia_final  = df_concorrencia_final.drop(columns = [ df_concorrencia.columns[k-1] ])

  
 

df_concorrencia_final = df_concorrencia_final.astype(float) 
#df_concorrencia_final = df_concorrencia_final.drop(columns=['ean','Produtos','Unit_Description','unit_ean_prod'])
df_concorrencia_final  = df_concorrencia_final.interpolate().bfill() 
df_concorrencia_final = df_concorrencia_final.interpolate(limit_direction='both')
df_concorrencia_final =  df_concorrencia_final.dropna(axis='columns') 
df_concorrencia_final = df_concorrencia_final.groupby( df_concorrencia_final.index).median()
df_concorrencia_final



# %% Df Prices Top Skus
# Df Prices Top SKus 
# Prices 
df_price_categories = df.copy() 
df_price_categories['Produtos'] = df_price_categories['Unit_Description'].astype(str) + " - " + df_price_categories['unit_ean'].astype(str)
df_price_categories = df_price_categories[df_price_categories['Unit_Description'].isin(df_top_skus['Unit_Description'].unique().tolist())]
df_price_categories = df_price_categories[['DateHour','order_id','order_item_id','Data','customer_id','Categoria','unit_ean','Unit_Description','Produtos','region_id','store_id','price_managers','Original_Price','Price']]
df_price_categories = df_price_categories.merge( df_users[['client_site_code','Region Name','pricing_group_id','size','Tipo_Cliente','tipo da loja','Não_Mercado','5-9 Cxs','1-4 Cxs']] ,how ='left', left_on='customer_id', right_on='client_site_code', suffixes=(False, False)).drop(columns=['client_site_code'])
#df_price_categories = df_price_categories[df_price_categories['Data'] >= pd.Timestamp('2021-03-01')]   
#df_price_categories = df_price_categories[df_price_categories['Data'] <= pd.Timestamp('2024-03-20')]   

df_price_categories['Flag_Store'] =  np.where((df_price_categories['store_id'].str.startswith('Estoque')  ) , 'Estoque'  , '-'  )
df_price_categories['Flag_Store'] =  np.where((df_price_categories['store_id'].str.startswith('Ofe')  ) , 'Ofertão'  , df_price_categories['Flag_Store']  )
df_price_categories['Flag_Store'] =  np.where((df_price_categories['Flag_Store'] ==  '-'  ) , '3P'  , df_price_categories['Flag_Store']  )

df_price_categories['order_item_id']= df_price_categories['order_item_id'].astype(np.int64).astype(str)
#df_price_categories  = df_price_categories[df_price_categories['order_item_id']=='593']

df_price_categories['unit_ean']= df_price_categories['unit_ean'].astype(np.int64).astype(str)
df_price_categories[df_price_categories['unit_ean'] == '7898215157403']

 
#df_price_categories['key_concorrencia'] = df_price_categories['Data'].astype(str)  +  df_price_categories['unit_ean'].astype(str) 
#df_price_categories = df_price_categories.merge(   df_concorrencia  ,how ='left', left_on='key_concorrencia', right_on='key_concorrencia', suffixes=(False, False))
 
 
# Prices 1-4 Cxs


df_prices_1_4_cxs  = df_price_categories.copy()
df_prices_1_4_cxs  = df_prices_1_4_cxs [df_prices_1_4_cxs['Flag_Store']=='Estoque']
 
 
df_prices_1_4_cxs  = df_prices_1_4_cxs [df_prices_1_4_cxs['1-4 Cxs']==1]
df_prices_1_4_cxs  = df_prices_1_4_cxs [df_prices_1_4_cxs['pricing_group_id'] == 70]
df_prices_1_4_cxs  = df_prices_1_4_cxs [df_prices_1_4_cxs['Não_Mercado'] == 0]
df_prices_1_4_cxs  = pd.pivot_table(df_prices_1_4_cxs , values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
 
df_prices_1_4_cxs.columns = df_prices_1_4_cxs .columns.droplevel(0)
df_prices_1_4_cxs.columns = df_prices_1_4_cxs .columns.droplevel(0)
df_prices_1_4_cxs  = df_prices_1_4_cxs .reset_index(drop = False).set_index('Data')

df_prices_1_4_cxs  = pd.DataFrame(df_prices_1_4_cxs .to_records()).set_index('Data')
df_prices_1_4_cxs  =  pd.DataFrame(df_prices_1_4_cxs .asfreq('D').index).set_index('Data').merge(df_prices_1_4_cxs , left_index = True, right_index=True,how = "left") 
#df_prices_1_4_cxs  = df_prices_1_4_cxs .interpolate().bfill() 

df_prices_1_4_cxs.columns=[ "Price BAU 1-4 Cxs - " + str(df_prices_1_4_cxs .columns[k-1]).title()  for k in range(1, df_prices_1_4_cxs .shape[1] + 1)]
 
 
# Prices 5-9 Cxs

df_prices_5_9_cxs = df_price_categories.copy()
df_prices_5_9_cxs = df_prices_5_9_cxs[df_prices_5_9_cxs['Flag_Store']=='Estoque']
df_prices_5_9_cxs = df_prices_5_9_cxs[df_prices_5_9_cxs['1-4 Cxs']==0]
df_prices_5_9_cxs = df_prices_5_9_cxs[df_prices_5_9_cxs['pricing_group_id'] == 70]
df_prices_5_9_cxs = df_prices_5_9_cxs[df_prices_5_9_cxs['Não_Mercado'] == 0]
df_prices_5_9_cxs = pd.pivot_table(df_prices_5_9_cxs, values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
df_prices_5_9_cxs.columns = df_prices_5_9_cxs.columns.droplevel(0)
df_prices_5_9_cxs.columns = df_prices_5_9_cxs.columns.droplevel(0)
df_prices_5_9_cxs = df_prices_5_9_cxs.reset_index(drop = False).set_index('Data')
df_prices_5_9_cxs = pd.DataFrame(df_prices_5_9_cxs.to_records()).set_index('Data')
 
df_prices_5_9_cxs =  pd.DataFrame(df_prices_5_9_cxs.asfreq('D').index).set_index('Data').merge(df_prices_5_9_cxs, left_index = True, right_index=True,how = "left") 
df_prices_5_9_cxs.columns=[ "Price BAU 5-9 Cxs " + str(df_prices_5_9_cxs.columns[k-1]).title()  for k in range(1, df_prices_5_9_cxs.shape[1] + 1)]
#df_prices_5_9_cxs = df_prices_5_9_cxs.interpolate().bfill() 

 
# Prices 1-4 Cxs Ofertão
  
df_prices_1_4_cxs_ofertao = df_price_categories.copy()
df_prices_1_4_cxs_ofertao= df_prices_1_4_cxs_ofertao[df_prices_1_4_cxs_ofertao['Flag_Store']=='Estoque']
df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao[df_prices_1_4_cxs_ofertao['1-4 Cxs']==1]
df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao[df_prices_1_4_cxs_ofertao['pricing_group_id'] == 70]
df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao[df_prices_1_4_cxs_ofertao['Não_Mercado'] == 0]
df_prices_1_4_cxs_ofertao = pd.pivot_table(df_prices_1_4_cxs_ofertao, values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
 
df_prices_1_4_cxs_ofertao.columns = df_prices_1_4_cxs_ofertao.columns.droplevel(0)
df_prices_1_4_cxs_ofertao.columns = df_prices_1_4_cxs_ofertao.columns.droplevel(0)
df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao.reset_index(drop = False).set_index('Data')

df_prices_1_4_cxs_ofertao = pd.DataFrame(df_prices_1_4_cxs_ofertao.to_records()).set_index('Data')
df_prices_1_4_cxs_ofertao =  pd.DataFrame(df_prices_1_4_cxs_ofertao.asfreq('D').index).set_index('Data').merge(df_prices_1_4_cxs_ofertao, left_index = True, right_index=True,how = "left") 
df_prices_1_4_cxs_ofertao.columns=[ "Price Ofertão 1-4 Cxs - " + str(df_prices_1_4_cxs_ofertao.columns[k-1]).title()  for k in range(1, df_prices_1_4_cxs_ofertao.shape[1] + 1)]
#df_prices_1_4_cxs _ofertao = df_prices_1_4_cxs _ofertao.interpolate().bfill() 

# Prices 5-9 Cxs

df_prices_5_9_cxs_ofertao = df_price_categories.copy()
df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao[df_prices_5_9_cxs_ofertao['Flag_Store']=='Estoque']
df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao[df_prices_5_9_cxs_ofertao['1-4 Cxs']==0]
df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao[df_prices_5_9_cxs_ofertao['pricing_group_id'] == 70]
df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao[df_prices_5_9_cxs_ofertao['Não_Mercado'] == 0]
df_prices_5_9_cxs_ofertao = pd.pivot_table(df_prices_5_9_cxs_ofertao, values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
df_prices_5_9_cxs_ofertao.columns = df_prices_5_9_cxs_ofertao.columns.droplevel(0)
df_prices_5_9_cxs_ofertao.columns = df_prices_5_9_cxs_ofertao.columns.droplevel(0)
df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao.reset_index(drop = False).set_index('Data')
df_prices_5_9_cxs_ofertao = pd.DataFrame(df_prices_5_9_cxs_ofertao.to_records()).set_index('Data')
df_prices_5_9_cxs_ofertao =  pd.DataFrame(df_prices_5_9_cxs_ofertao.asfreq('D').index).set_index('Data').merge(df_prices_5_9_cxs_ofertao, left_index = True, right_index=True,how = "left") 
df_prices_5_9_cxs_ofertao.columns=[ "Price Ofertão 5-9 Cxs " + str(df_prices_5_9_cxs_ofertao.columns[k-1]).title()  for k in range(1, df_prices_5_9_cxs_ofertao.shape[1] + 1)]
#df_prices_5_9_cxs_ofertao = df_prices_5_8_cxs_mercado.interpolate().bfill() 


df_price_mediana = df_price_categories.copy()
df_price_mediana = pd.pivot_table(df_price_mediana, values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
df_price_mediana = df_price_mediana.interpolate().bfill() 
df_price_mediana =  df_price_mediana.interpolate(limit_direction='both')
df_price_mediana.columns = df_price_mediana.columns.droplevel(0)
df_price_mediana.columns = df_price_mediana.columns.droplevel(0)
df_price_mediana = df_price_mediana.reset_index(drop = False).set_index('Data')
df_price_mediana = pd.DataFrame(df_price_mediana.to_records()).set_index('Data')
df_price_mediana.columns=[ "Price " +   str(df_price_mediana.columns[k-1]) for k in range(1, df_price_mediana.shape[1] + 1)]

df_price_final = df_price_mediana.merge(df_prices_1_4_cxs, how = 'left', left_index = True, right_index=True)
df_price_final = df_price_final.merge(df_prices_5_9_cxs, how = 'left', left_index = True, right_index=True) 
df_price_final = df_price_final.merge(df_prices_1_4_cxs_ofertao, how = 'left', left_index = True, right_index=True) 
df_price_final = df_price_final.merge(df_prices_5_9_cxs_ofertao, how = 'left', left_index = True, right_index=True) 
df_price_final = df_price_final.astype(float)
df_price_final = df_price_final.replace(np.nan,0)
df_price_final


# Ajusta os preços de 1-4 e 5-9 para quem tava sem preço e ajusta ofertão  
 
for sku in top_skus: 
  sku_cols = df_price_final.columns[df_price_final.columns.str.endswith(sku)].tolist()

#  sku_cols_ofertao_1_4 = df_price_final.columns[df_price_final.columns.str.startswith('Price Ofertão 1-4 Cxs')].tolist()
#  sku_cols_ofertao_5_9 = df_price_final.columns[df_price_final.columns.str.startswith('Price Ofertão 5-9 Cxs')].tolist()

 
  for col in range(0, len(sku_cols)):  
    # Acerta Ofertão para Preço Médio
    if col ==0: 
      
      sku_cols_ofertao = df_price_final[sku_cols].columns[df_price_final[sku_cols].columns.str.startswith('Price Ofertão 1-4 Cxs')].tolist()    
      
      if len(sku_cols_ofertao)>0:
        df_price_final[sku_cols[col]] =  np.where(df_price_final[sku_cols_ofertao[0]] > 0  , df_price_final[sku_cols_ofertao[0]] ,  df_price_final[sku_cols[col]] )

    # Acerta Preços caso n tenha venda e acerta ofertão caso tenha nos 1-4 e nos 5-9        

    elif col >0: 
      
 
      tipo_sku = sku_cols[col][:13]
      if tipo_sku != 'Price Ofertão':
 
        df_price_final[sku_cols[col]] =  np.where(df_price_final[sku_cols[col]]  == 0  , df_price_final[sku_cols[0]]  ,  df_price_final[sku_cols[col]] )
 
 
        if tipo_sku[len(tipo_sku)-3:]  == '1-4': 
          sku_cols_ofertao = df_price_final[sku_cols].columns[df_price_final[sku_cols].columns.str.startswith('Price Ofertão 1-4 Cxs')].tolist()    
          if len(sku_cols_ofertao)>0:

                  
            df_price_final[sku_cols[col]] =  np.where(df_price_final[sku_cols_ofertao[0]] > 0  , df_price_final[sku_cols_ofertao[0]] ,  df_price_final[sku_cols[col]] )
            
            #  print('aqui')
            #  print(sku_cols_ofertao[0])

        elif tipo_sku[len(tipo_sku)-3:] == '5-9':

          sku_cols_ofertao = df_price_final[sku_cols].columns[df_price_final[sku_cols].columns.str.startswith('Price Ofertão 5-9 Cxs')].tolist()    

          if len(sku_cols_ofertao)>0:       

            df_price_final[sku_cols[col]] =  np.where(df_price_final[sku_cols_ofertao[0]]  > 0  , df_price_final[sku_cols_ofertao[0]]  ,  df_price_final[sku_cols[col]] )

 
cols_df_prices_1_4_cxs = df_prices_1_4_cxs.columns.to_list()
cols_df_prices_5_9_cxs = df_prices_5_9_cxs.columns.to_list() 
cols_df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao.columns.to_list() 
cols_df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao.columns.to_list() 
cols_df_prices_mediana = df_price_mediana.columns.to_list() 
df_price_final = df_price_final.drop(columns= cols_df_prices_1_4_cxs_ofertao + cols_df_prices_5_9_cxs_ofertao )
df_price_final



#df_price_final[df_price_final.columns[df_price_final.columns.str.endswith('7891107101621')].tolist()]   # Aqui pode # Filtrar
 

# %% Df Skus 
# DF SKUS 
df_skus = df.copy()  

#df_skus = df_skus[df_skus['Data'] > pd.Timestamp('2024-01-01')]  
#df_skus[['Categoria','Gmv']].groupby(['Categoria']).sum().sort_values('Gmv' , ascending =  False)  
df_skus['key'] = df_skus['Data'].astype(str) + df_skus['customer_id'].astype(str)
df_skus = df_skus[df_skus['unit_ean_prod'].isin(top_skus)]
df_skus['Produto'] = 'Gmv Categoria ' + df_skus['Categoria'].astype(str) + ' ' + df_skus['Unit_Description'].astype(str) + ' - ' +   df_skus['unit_ean_prod'].astype(str)
df_skus = df_skus[['key','Produto','Gmv']]
df_skus = pd.pivot_table(df_skus, values=['Gmv'], index=['key'] , columns=['Produto'],aggfunc={ 'Gmv': [ "sum" ]})
df_skus.columns = df_skus .columns.droplevel(0)
df_skus.columns = df_skus .columns.droplevel(0)
df_skus  = df_skus .reset_index(drop = False).set_index('key')
df_skus  = pd.DataFrame(df_skus .to_records()).set_index('key')
 

for k in range(1, df_skus.shape[1] + 1):
      
  df_skus['Positivação ' + df_skus.columns[k-1][3:]] = np.where(( df_skus.iloc[:,k-1:k]  > 0 )  ,   1  , 0  )
 

df_skus = df_skus.replace(np.nan, 0 )
df_skus = df_skus.reset_index(drop = False)
 
df_skus



# %% Df Categorias 
# Df Categoria

df_categoria = df.copy()  
#df_categoria = df_categoria[df_categoria['unit_ean_prod']== '7896079500151']
#df_categoria = df_categoria[df_categoria['Data'] <  pd.Timestamp('2024-02-15')]   
#df_categoria = df_categoria[df_categoria['Data'] >  pd.Timestamp('2024-02-07')]   

 
#df_categoria = df_categoria[df_categoria['Data'] > pd.Timestamp('2024-01-01')]  
#df_categoria[['Categoria','Gmv']].groupby(['Categoria']).sum().sort_values('Gmv' , ascending =  False)  
df_categoria['key'] = df_categoria['Data'].astype(str) + df_categoria['customer_id'].astype(str)
df_categoria['Pedidos'] = df_categoria['key'] 
df_categoria['Positivação'] = df_categoria['Data'].astype(str) + df_categoria['customer_id'].astype(str) + df_categoria['Categoria'].astype(str)
df_categoria = df_categoria[df_categoria['Categoria'].isin(lista_categorias)]
 
df_categoria = df_categoria.groupby( ['key','Categoria']).agg({'Gmv':'sum', 'Quantity':'sum','Price':'mean', 'unit_ean_prod': 'nunique'  }).reset_index(drop= False)
df_categoria = df_categoria.reset_index(drop = 'False')
df_categoria= df_categoria.rename(columns = {'customer_id':'User','unit_ean_prod':'Produtos'})
df_categoria_dummies = pd.get_dummies(df_categoria['Categoria'] ).astype(float) 
df_categoria= pd.get_dummies(df_categoria, columns = ['Categoria'] , drop_first = False)


df_categoria.iloc[:,1:] = df_categoria.iloc[:,1:].astype(float)
df_calcula_categorias = df_categoria.copy() 
df_calcula_categorias
 
 
for i in range(1, df_categoria.iloc[:,5:].shape[1]+1): 
    col_categoria = df_categoria.columns[4+i]  
    df_calcula_categorias['Gmv ' + col_categoria ] = df_categoria[col_categoria] * df_categoria['Gmv']
    df_calcula_categorias['Positivação ' + col_categoria ] = df_categoria[col_categoria]  
    df_calcula_categorias['Quantity ' + col_categoria ] = df_categoria[col_categoria] * df_categoria['Quantity']
    df_calcula_categorias['Price ' + col_categoria ] = df_categoria[col_categoria] * df_categoria['Price']
    df_calcula_categorias['Top Produtos ' + col_categoria ] = df_categoria[col_categoria] * df_categoria['Produtos']
    df_calcula_categorias = df_calcula_categorias.drop(columns = [col_categoria])


df_calcula_categorias 

df_categoria = df_categoria.merge( df_categoria_dummies ,how ='left', left_index= True, right_index=True, suffixes=(False, False))
 

df_categoria = df_calcula_categorias.copy()
 
cols_positivacao = df_categoria.columns[df_categoria.columns.str.startswith('Positivação')].tolist()
cols_price = df_categoria.columns[df_categoria.columns.str.startswith('Price')].tolist()

cols_top_produtos = df_categoria.columns[df_categoria.columns.str.startswith('Top Produtos')].tolist()


cols_price.remove('Price')


df_categoria_pos = df_categoria[['key'] + cols_positivacao].groupby('key').max()
df_categoria_price= df_categoria[cols_price + ['key']].groupby('key').mean()
df_categoria_gmv_qtd = df_categoria.drop(columns = ['Gmv','Quantity','Price'] + cols_positivacao + cols_price ).groupby('key').sum()
df_categoria = df_categoria_gmv_qtd.merge( df_categoria_pos ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
df_categoria = df_categoria.merge( df_skus ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
#df_categoria = df_categoria.merge( df_categoria_price ,how ='left', left_on='key', right_on='key', suffixes=(False, False))


df_categoria = df_categoria.reset_index(drop = False)
df_categoria = df_categoria.replace(np.nan, 0)
cols_positivacao = df_categoria.columns[df_categoria.columns.str.startswith('Positivação')].tolist()


cols_gmv_categoria = df_categoria.columns[df_categoria.columns.str.startswith('Gmv')].tolist()

cols_gmv_categoria
 
# %% Df Grouped
# Df Grouped

df_grouped = df.copy() 

 
df_grouped['key'] = df_grouped['Data'].astype(str) + df_grouped['customer_id'].astype(str)
df_grouped = df_grouped.rename(columns = {'customer_id':'User'})
df_grouped = pd.concat([df_grouped[['Data','key','User']].groupby(['key','User']).max().reset_index(drop=False),df_trafego[['Data','key','User']].groupby(['key','User']).max().reset_index(drop=False)])
df_grouped = df_grouped.groupby('key').max().reset_index(drop = False)
df_grouped = df_grouped.merge( df_users[['client_site_code','Region Name','region_id']] ,how ='left', left_on='User', right_on='client_site_code', suffixes=(False, False))
df_grouped = df_grouped.drop(columns=['client_site_code'])
df_grouped = df_grouped.merge(df_trafego.drop(columns = ['Data','User']) ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
df_grouped = df_grouped.merge( df_vendas ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
df_grouped = df_grouped.merge( df_categoria ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
df_grouped['key_categoria'] = 1
df_grouped = df_grouped.merge( df_top_categorias ,how ='left', left_on='key_categoria', right_on='key_categoria', suffixes=(False, False))
df_grouped = df_grouped.replace(np.nan, 0)

df_grouped = df_grouped.drop(columns = ['key_categoria'])
df_grouped
#df_grouped[]  = df_modelo[cols_price_clubbi + cols_price_concorrencia].interpolate().bfill() 
#df_grouped[] = df_modelo[cols_price_clubbi + cols_price_concorrencia].interpolate(limit_direction='both')

#df_grouped['Gmv Categoria_Leite'] = df_grouped['Gmv Categoria_Leite']replace(np.nan, 0)

#df_grouped['Gmv Categoria_Leite'] = df_grouped['Gmv Categoria_Leite'].interpolate().bfill() 


#'Gmv Categoria_Leite'


# %% Df Modelo
# Df Modelo

df_modelo = df_grouped.copy()  
df_modelo = df_modelo.merge( df_users[['client_site_code','size_final','1-4 Cxs']] ,how ='left', left_on='User', right_on='client_site_code', suffixes=(False, False)).drop(columns=['client_site_code'])
df_modelo = df_modelo[df_modelo['1-4 Cxs']==1]

#df_modelo = df_modelo[df_modelo['Unit']==1]
 

#df_modelo = df_modelo[df_modelo['size_final']!='3-4 caixas']
df_modelo =  df_modelo.drop(columns = ['1-4 Cxs','size_final'])

df_modelo = df_modelo[df_modelo['Region Name']=='RJC']
 
if region !=0: df_modelo = df_modelo[df_modelo['region_id'] == region ]
#print(df_modelo['region_id'].unique())
  
   
df_modelo = df_modelo[df_modelo['Data']>='2021-01-01']  
df_modelo = df_modelo[df_modelo['Data']<data_final] 
df_modelo = df_modelo.set_index('Data')  

df_correl_teste = df_modelo.copy()
df_modelo = df_modelo.drop(columns= ['key','User','Region Name','region_id'])
df_modelo.replace(np.nan, 0)
 
 
df_modelo = df_modelo.groupby( df_modelo.index).sum()
 
cols_sem_price = df_modelo.columns.to_list()

#cols_df_prices_mediana + cols_df_prices_1_4_cxs + cols_df_prices_5_9_cxs


df_modelo = df_modelo.merge(df_price_final, left_index = True, right_index=True,how = "left")  
 
 
df_modelo = df_modelo.merge(df_concorrencia_final, left_index = True, right_index=True,how = "left") 
cols_price_concorrencia = df_concorrencia_final.columns.to_list()

cols_price_clubbi = df_price_final.columns.to_list()
 

# Métricas Trafego
 
df_modelo['conversao_trafego'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego'] ) ,   np.nan  , df_modelo['Pedidos'] /df_modelo['Trafego']  )
df_modelo['conversao_trafego'] = df_modelo['conversao_trafego'].interpolate(limit_direction='both')

df_modelo['conversao_trafego_search'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Search_Products'] ) ,   np.nan  , df_modelo['Pedidos'] /df_modelo['Trafego_Search_Products']  )
df_modelo['conversao_trafego_search'] = df_modelo['conversao_trafego_search'].interpolate(limit_direction='both')


df_modelo['conversao_trafego_add'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Add_To_Cart'] ) ,   np.nan  , df_modelo['Pedidos'] /df_modelo['Trafego_Add_To_Cart']  )
df_modelo['conversao_trafego_add'] = df_modelo['conversao_trafego_add'].interpolate(limit_direction='both')

df_modelo['conversao_trafego_checkout'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Checkout'] ) ,   np.nan  , df_modelo['Pedidos'] /df_modelo['Trafego_Checkout']  )
df_modelo['conversao_trafego_checkout'] = df_modelo['conversao_trafego_add'].interpolate(limit_direction='both')


df_modelo['Trafego'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego'] ) ,   df_modelo['Pedidos']/df_modelo['conversao_trafego'] , df_modelo['Trafego']  ).astype(int)
df_modelo['Trafego_Search_Products'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Search_Products'] ) ,   df_modelo['Pedidos']/df_modelo['conversao_trafego_search']  , df_modelo['Trafego_Search_Products']  ).astype(int)
df_modelo['Trafego_Add_To_Cart'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Add_To_Cart'] ) ,   df_modelo['Pedidos']/df_modelo['conversao_trafego_add']  , df_modelo['Trafego_Add_To_Cart']  ).astype(int)
df_modelo['Trafego_Checkout'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Checkout'] ) ,  df_modelo['Pedidos'] / df_modelo['conversao_trafego_checkout']  , df_modelo['Trafego_Checkout']  ).astype(int)

df_modelo = df_modelo.drop(columns= ['conversao_trafego','conversao_trafego_search','conversao_trafego_add','conversao_trafego_checkout'])

 
df_modelo =  pd.DataFrame(df_modelo.asfreq('D').index).set_index('Data').merge(df_modelo, left_index = True, right_index=True,how = "left") 

  
# Df Futuro 
 
future_dates = pd.date_range(df_modelo.index[-1], freq = 'D', periods = 2)
future_dates = pd.DataFrame(future_dates, columns=['Data'])
future_dates = future_dates.set_index('Data')
future_dates['Gmv'] = np.nan
future_dates = future_dates.iloc[1:,:]
future_dates
 
# Df Modelo Final

df_modelo = pd.concat([df_modelo, future_dates])  



df_modelo[cols_price_clubbi + cols_price_concorrencia]  = df_modelo[cols_price_clubbi + cols_price_concorrencia].interpolate().bfill() 
df_modelo[cols_price_clubbi + cols_price_concorrencia] = df_modelo[cols_price_clubbi + cols_price_concorrencia].interpolate(limit_direction='both')
 
df_modelo = df_modelo.merge(df_eventos, left_index = True, right_index=True,how = "left") 
df_modelo = df_modelo.merge(  df_ofertao[ofertao_columns], how='left', left_index=True, right_index=True)   
df_modelo = df_modelo.replace(np.nan,0)
 
df_modelo = df_modelo.dropna()  
  
df_modelo = df_modelo.sort_index(ascending=True)

df_modelo['Gmv_Shift'] = df_modelo['Gmv'].shift(periods=  70, freq="D")
df_modelo['Quantity_Shift'] = df_modelo['Quantity'].shift(periods=  70, freq="D")
df_modelo['Pedidos_Shift'] = df_modelo['Pedidos'].shift(periods=   70, freq="D")
df_modelo['Trafego_Shift'] = df_modelo['Trafego'].shift(periods=   70, freq="D")



df_modelo['Gmv'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,   df_modelo['Gmv_Shift']  ,df_modelo['Gmv'] )
df_modelo['Quantity'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Quantity_Shift']   , df_modelo['Quantity'] )
df_modelo['Pedidos'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Pedidos_Shift']  , df_modelo['Pedidos'] ) 
df_modelo['Trafego'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Trafego_Shift']  , df_modelo['Trafego'] ) 


df_modelo = df_modelo.drop(columns=['Gmv_Shift','Quantity_Shift','Pedidos_Shift','Trafego_Shift'])
df_modelo
 


# Não Me xer Nessa Parte * Atenção 
df_delta_prices =  df_modelo[cols_price_clubbi + cols_price_concorrencia]
#df_delta_prices = df_delta_prices[df_delta_prices.columns[df_delta_prices.columns.str.endswith('7891107101621')].tolist()]   # Aqui pode # Filtrar
df_delta_final = df_delta_prices.copy() 
df_delta_final['Flag'] = 1  
df_delta_final = df_delta_final[['Flag']]
df_delta_final
  

# Não Me xer Nessa Parte * Atenção 
 
# Lags, Windowns, Deltas, Preços Clubbi e Concorrentes 
 
for k in range(0, len(top_skus)):  
    # Loop SKU
 
  prod = top_skus[k]  
  cols_produto = df_delta_prices.columns[df_delta_prices.columns.str.endswith(prod)].tolist()
  
  df_produto_prices = df_delta_prices[cols_produto]
  cols_clubbi = df_produto_prices.columns[df_produto_prices.columns.str.startswith('Price')].tolist() 
  
  
  cols_prices_concorrencia = df_produto_prices.drop(columns=cols_clubbi).columns.tolist()
  flag_conc = len(cols_prices_concorrencia)


      
   
  cols_positivacao = df_modelo.columns[df_modelo.columns.str.endswith(prod)].tolist()
  cols_positivacao = df_modelo[cols_positivacao].columns[df_modelo[cols_positivacao].columns.str.startswith('Positivação')].tolist()

 
  df_modelo['% Share ' + cols_positivacao[0]] =  np.where((df_modelo['Pedidos'] == 0 ) , 0 , df_modelo[cols_positivacao[0]]/ df_modelo['Pedidos']  )
 
    

  
#  delta_clubbi = 'Positivação Top Sku ' + price_clubbi      
#  df_delta_final[delta_clubbi] = (df_delta_final['Lag 7 ' + price_clubbi ]  +  df_delta_final['Lag 14 ' + price_clubbi ])/2
  


  for c in range(0, len(cols_clubbi)):    

#    price_clubbi =  cols_clubbi[c][:17][6:]           
    price_clubbi =  cols_clubbi[c]
 
    lag_list = [1,5,7,14,21,28]
    
    # Lags e Delta Lag 
      
    for lag in lag_list:

      # Lag  
      delta_clubbi = 'Lag ' + str(lag) +  ' ' + price_clubbi 
      df_delta_final[delta_clubbi] = df_delta_prices[cols_clubbi[c]].shift(periods=  lag, freq="D") 
      # Delta Lag
      delta_clubbi = 'Delta Lag ' + str(lag) +  ' ' +  price_clubbi 
      df_delta_final[delta_clubbi] = (df_delta_prices[cols_clubbi[c]]/ df_delta_prices[cols_clubbi[c]].shift(periods=  lag, freq="D"))-1


    # delta_clubbi = 'Delta Lag 1 '   +  price_clubbi 

    # delta_clubbi_percent = 'Delta Wind 1 Redução 0-5% ' + price_clubbi       
    # df_delta_final[delta_clubbi_percent] = np.where((df_delta_final[delta_clubbi] >-0.20) &  (df_delta_final[delta_clubbi]<0) , 1 , 0 )

    # delta_clubbi_percent = 'Delta Wind 1 Redução 0,5% ' + price_clubbi       
    # df_delta_final[delta_clubbi_percent] = np.where((df_delta_final[delta_clubbi]<-0.05) , 1 , 0 )




    # Média dos Lags 
      
    delta_clubbi = 'Lag- Mean 7/14 ' + price_clubbi      
    df_delta_final[delta_clubbi] = (df_delta_final['Lag 7 ' + price_clubbi ]  +  df_delta_final['Lag 14 ' + price_clubbi ])/2

    delta_clubbi = 'Lag- Mean 7/14/21 ' + price_clubbi  
    df_delta_final[delta_clubbi] = (df_delta_final['Lag 7 ' + price_clubbi ]  +  df_delta_final['Lag 14 ' + price_clubbi ] +  df_delta_final['Lag 21 ' +  price_clubbi ]      )/3
    
    delta_clubbi = 'Lag- Mean 7/14/21/28 ' +  price_clubbi     
    df_delta_final[delta_clubbi] = (df_delta_final['Lag 7 ' + price_clubbi]  +  df_delta_final['Lag 14 ' + price_clubbi ] +  df_delta_final['Lag 21 ' +  price_clubbi ]  +  df_delta_final['Lag 28 ' + price_clubbi ]  )/4 

    # Média Windown 
    delta_clubbi = 'Wind 2D ' + price_clubbi     
    df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(2).mean()
    delta_clubbi = 'Wind 5D ' + price_clubbi     
    df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(5).mean()
    delta_clubbi = 'Wind 7D ' + price_clubbi     
    df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(7).mean()
    delta_clubbi = 'Wind 14D ' + price_clubbi     
    df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(14).mean()
    delta_clubbi = 'Wind 21D ' + price_clubbi     
    df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(21).mean()
    delta_clubbi = 'Wind 28D ' + price_clubbi     
    df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(28).mean()
    
    # Delta Média Lags  
    delta_clubbi_var = 'Lag- Mean 7/14 ' + price_clubbi         
    delta_clubbi = 'Delta Lag- Mean 7/14 ' + price_clubbi       
    df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1 




    # delta_clubbi = 'Delta Lag- Mean 7/14/21 ' + price_clubbi       
    # df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1


    # delta_clubbi = 'Delta Lag- Mean 7/14/21/28 ' + price_clubbi       
    # df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

     

    # Delta Média Wind
    delta_clubbi_var = 'Wind 2D ' + price_clubbi         
    delta_clubbi = 'Delta Wind 2D ' + price_clubbi       
    df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

    delta_clubbi_var = 'Wind 5D ' + price_clubbi         
    delta_clubbi = 'Delta Wind 5D ' + price_clubbi       
    df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

    # delta_clubbi_var = 'Wind 7D ' + price_clubbi         
    # delta_clubbi = 'Delta Wind 7D ' + price_clubbi       
#    df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

    delta_clubbi_var = 'Wind 14D ' + price_clubbi         
    delta_clubbi = 'Delta Wind 14D ' + price_clubbi       
    df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

    

    delta_clubbi_var = 'Wind 21D ' + price_clubbi         
    delta_clubbi = 'Delta Wind 21D ' + price_clubbi       
    df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

    

    delta_clubbi_var = 'Wind 28D ' + price_clubbi         
    delta_clubbi = 'Delta Wind 28D ' + price_clubbi       
    df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1
  

    # delta_clubbi_percent = 'Delta Wind 28D Redução 0-5% ' + price_clubbi       
    # df_delta_final[delta_clubbi_percent] = np.where((df_delta_final[delta_clubbi] >-0.20) &  (df_delta_final[delta_clubbi]<0) , 1 , 0 )

    # delta_clubbi_percent = 'Delta Wind 28D Redução 0,5% ' + price_clubbi       
    # df_delta_final[delta_clubbi_percent] = np.where((df_delta_final[delta_clubbi]<-0.05) , 1 , 0 )

 

    
  #  df_delta_final[delta] =   df_delta_prices.apply(lambda row: ((row[cols_clubbi[z]] /row[concorrente_var])-1), axis=1) 
  #  df_delta_final[delta_clubbi] =   df_delta_prices.apply(lambda row: ((row[cols_clubbi[c]].shift (periods= 1  , freq="D") )), axis=1) 

  # Checa se tem concorrente 

    if flag_conc >0:
          
      df_conc = df_produto_prices[cols_prices_concorrencia]  
  # Se tem concorrente loop os concorrentes 

      for j in range(1,   df_conc.shape[1] +1):
      # Loop Concorrente 
        
        concorrente_var = df_delta_prices[cols_prices_concorrencia].columns[j-1] 
        for z in range(0, len(cols_clubbi)):    

  #        preco_clubbi =  cols_clubbi[z][:17][6:]         
          preco_clubbi =  cols_clubbi[z][:17][6:] 
          delta = 'Delta Concorrencia ' +  preco_clubbi + ' ' + concorrente_var  
          df_delta_final[delta] =   df_delta_prices.apply(lambda row: ((row[cols_clubbi[z]] /row[concorrente_var])-1), axis=1)

# Lags, Windowns, Deltas, Preços Clubbi e Concorrentes 
        
  
df_delta_final = df_delta_final.drop(columns=['Flag']) 
col_delta_prices = df_delta_final.columns.to_list()   
lag_lista = df_delta_final.columns[df_delta_final.columns.str.startswith('Lag')].tolist()   
mean_lista = df_delta_final.columns[df_delta_final.columns.str.startswith('Lag-')].tolist()   
wind_lista = df_delta_final.columns[df_delta_final.columns.str.startswith('Wind 7D')].tolist()   
delta_wind = df_delta_final.columns[df_delta_final.columns.str.startswith('Delta Wind')].tolist()    
delta_lag_mean = df_delta_final.columns[df_delta_final.columns.str.startswith('Delta Lag-')].tolist()   
delta_lag = df_delta_final.columns[df_delta_final.columns.str.startswith('Delta Lag')].tolist()   
delta_conc = df_delta_final.columns[df_delta_final.columns.str.startswith('Delta Conco')].tolist()   
#delta_conc = [i for i in  delta_conc if i not in delta_wind ] 
#delta_conc = [i for i in delta_conc if i not in delta_lag] 
  
  
price_lista = df_modelo.columns[df_modelo.columns.str.startswith('Price')].tolist()   
lista_trafego = df_modelo.columns[df_modelo.columns.str.startswith('Trafego')].tolist() 
lista_share_positivacao = df_modelo.columns[df_modelo.columns.str.startswith('% Share Positiv')].tolist() 



#df_modelo = df_modelo.drop(columns =  cols_price_concorrencia).merge(df_delta_final[delta_wind + delta_lag], left_index = True, right_index=True,how = "left") 
df_modelo = df_modelo.merge(df_delta_final[ delta_lag_mean + delta_wind + delta_conc], left_index = True, right_index=True,how = "left") 
df_correl_guarda = df_modelo.copy() 

 
delta_conc = df_delta_final.columns[df_delta_final.columns.str.startswith('Delta Conco')].tolist()   

df_modelo


# %% Correl 

 
# df_correl_teste = df_modelo.copy() 
 
# price_lista = df_modelo.columns[df_modelo.columns.str.startswith('Price')].tolist()

# lista_sku =  df_correl_teste.columns[df_correl_teste.columns.str.endswith('7896079500151')].tolist()   

# df_correl_teste = df_correl_teste[lista_sku]
# df_correl_teste

 
# categoria_quantity = []
# categoria_gmv = []
# categoria_positivacao = []
# categoria_precos = []
# categoria_precos_delta = []
# categoria_ofertao = []
# categoria_precos_delta_wind= []
# categoria_precos_delta_lag = []
# categoria_conc = []

# lista_categorias = list(map(lambda x: x.replace('Óleos, Azeites E Vinagres', 'Óleo'),lista_categorias ))
 


# for categoria in lista_categorias:
#   cate = categoria  

#   for lista in df_correl_teste.columns.to_list():
# #    print(lista)
#     if lista.find(cate, 0)>=0: 
#       if lista.find('% Share Positivação',  0 )>=0: categoria_positivacao = categoria_positivacao  + [lista] 
#       elif lista.find('Quantity',  0 )>=0: categoria_quantity = categoria_quantity   + [lista] 
#       elif lista.find('Gmv',  0 )>=0: categoria_gmv = categoria_gmv  + [lista] 
#       elif lista.find('Price BAU 1-4',  0 )>=0: categoria_precos = categoria_precos  + [lista]
#       elif lista.find('Delta Wind ',  0 )>=0: categoria_precos_delta_wind = categoria_precos_delta_wind  + [lista]
#       elif lista.find('Delta Lag ',  0 )>=0: categoria_precos_delta_lag = categoria_precos_delta_lag  + [lista]
#       elif lista.find('Delta Conc',  0 )>=0: categoria_conc = categoria_precos_delta_lag  + [lista]
#       elif lista.find('Ofertão',  0 )>=0: categoria_ofertao = categoria_ofertao  + [lista]


# categoria_quantity = list(dict.fromkeys(categoria_quantity))
# categoria_gmv = list(dict.fromkeys(categoria_gmv))
# categoria_positivacao = list(dict.fromkeys(categoria_positivacao))
# categoria_precos = list(dict.fromkeys(categoria_precos))
# categoria_precos_delta_wind = list(dict.fromkeys(categoria_precos_delta_wind))
# categoria_precos_delta_lag = list(dict.fromkeys(categoria_precos_delta_lag))
# categoria_precos_delta = list(dict.fromkeys(categoria_precos_delta))
# categoria_ofertao = list(dict.fromkeys(categoria_ofertao))
# categoria_conc = list(dict.fromkeys(categoria_conc))

  
# #df_correl_teste = df_correl_teste[categoria_gmv + categoria_positivacao   + categoria_precos + categoria_conc]
# #df_correl_teste


   
# cols_price_clubbi_5_9 = [i for i in df_correl_teste.columns.to_list() if i.find('5-9 Cxs' ,0)>=0] 
# cols_conco_outras = [i for i in df_correl_teste.columns.to_list() if i.find('Delta Concorrencia L' ,0)>=0] 
# #cols_outras_price = [i for i in df_correl_teste.columns.to_list() if i.find('Price L' ,0)>=0] 
  
 

# #col = 'Positivação  Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151'

# col = 'Gmv Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151'
# #col = '% Share Positivação  Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151' 
# #col = 'Positivação  Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151'

# val = 0.2
# df_correl_teste  =  df_correl_teste[df_correl_teste[col] > 0 ]

# df_correl_teste  =  df_correl_teste[df_correl_teste.index >= pd.Timestamp('2024-02-21') ]
# df_correl_teste  =  df_correl_teste[df_correl_teste.index < data_final ]
# df_correl_teste  =  df_correl_teste[df_correl_teste.index <  pd.Timestamp('2024-03-26') ]
# df_correl_teste = df_correl_teste.drop(columns= cols_price_clubbi_5_9)
# df_correl_teste = df_correl_teste.drop(columns= cols_conco_outras)
# #df_correl_teste = df_correl_teste.drop(columns= cols_outras_price)


# #df_correl_teste.columns.to_list() 
 
# df_correl_teste = df_correl_teste[df_correl_teste[col]>df_correl_teste[col].quantile(val)]
# df_correl_teste[df_correl_teste[col]<df_correl_teste[col].quantile(val)].sort_values(col, ascending= False)
 

# fig, ax = plt.subplots(figsize=(2,8))        
# correlation_matrix = df_correl_teste.reset_index(drop=True).corr() 
 
# # sns.heatmap(correlation_matrix , annot=True, linewidths=.5, ax=ax) 
# # plt.title("Correlation Categoria")
# # plt.ylabel("Features") 
# # plt.show()
 
 
# # Heatmap Categoria


# #fig, ax = plt.subplots(figsize=(,60))        
# correlation_category = correlation_matrix.copy()
# correlation_category = correlation_matrix.iloc[:,8:9].sort_values(correlation_matrix.iloc[:,8:9].columns.to_list()[0],ascending = False )
# #correlation_category = correlation_matrix.iloc[:,0:1].sort_values(correlation_matrix.iloc[:,0:1].columns.to_list()[0],ascending = False )
# sns.heatmap(correlation_category , annot=True, linewidths=.5, ax=ax) 
# plt.title("Correlation Categoria")
# plt.ylabel("Features") 
# plt.show()

 

# correlation_category.index.to_list()

# #df_teste = df_teste[df_teste['Price']>df_teste["Price"].quantile(0.01)]

# #df_correl

# #categoria_positivacao
# #categoria_quantity
# #categoria_precos


# # df_guarda = df_modelo.copy()
# # df_guarda = df_guarda.T
# # df_guarda = df_guarda.reset_index(drop = False).iloc[:,0:1]
# # df_guarda['key'] = df_guarda['index']
# # df_guarda.groupby('key').count().sort_values('index',ascending = True)





# # %% Plot Prices
 
 
# #df_plot_prices = df_correl_teste[categoria_gmv + cols_price_clubbi].iloc[:,0:5][df_correl_teste.index > pd.Timestamp('2024-02-20')]
# #df_plot_prices
# # % %
# #df_plot_prices = df_plot_prices[df_plot_prices.columns[df_plot_prices.columns.str.endswith('7896079500151')].tolist()]
# df_plot_prices = df_correl_teste.copy() 

# print( df_correl_teste.columns.to_list())
  


# # %% Plotly

# import plotly.express as py 

# #import plotly.graph_objs as go

# dados_x= df_plot_prices.index
# dados_y= df_plot_prices['Price BAU 1-4 Cxs - Leite Uht Integral Elegê Caixa 1L - 7896079500151'] 

# dados_y2= df_plot_prices['% Share Positivação  Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151']

# dados_y3= df_plot_prices['Delta Concorrencia BAU 1-4 Cxs Torre_Cia Leite UHT Integral Elegê Caixa 1l - 7896079500151']


# dados_y4= df_plot_prices['Gmv Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151']
 

# dados_y5= df_plot_prices['Torre_Cia Leite UHT Integral Elegê Caixa 1l - 7896079500151']


# fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Data", y="Preço")  , markers = True, title="Price", height=400, width=1000, line_shape='spline')
 

# fig.show()



# fig=py.line(x=dados_x, y=dados_y2,  labels=dict(x="Data", y="% Share Positivação")  , markers = True, title="% Share Positivação", height=400, width=1000, line_shape='spline')


# fig.show()


# fig=py.line(x=dados_x, y=dados_y5,  labels=dict(x="Data", y="Delta Preço Concorrencia")  , markers = True, title="Delta Preço Concorrencia Torre_Cia", height=400, width=1000, line_shape='spline')


# fig.show()



# fig=py.line(x=dados_x, y=dados_y3,  labels=dict(x="Data", y="Delta Preço Concorrencia")  , markers = True, title="Delta Preço Concorrencia Torre_Cia", height=400, width=1000, line_shape='spline')


# fig.show()



# fig=py.line(x=dados_x, y=dados_y4,  labels=dict(x="Data", y="Gmv")  , markers = True, title="Gmv", height=400, width=1000, line_shape='spline')


# fig.show()

# COLOR_TEMPERATURE = "#7900F1"
# COLOR_PRICE = "#CD4C46"

 
# fig, ax = plt.subplots(figsize=(12,5))
# ax2 = ax.twinx() 
# ax.plot(dados_x ,dados_y ,label =  'Price'  , color=COLOR_TEMPERATURE) 
# #ax.plot(df_plot.index , df_plot[var_predicao + '_window_28D_mean'],label =  var_predicao  + ' Média 28 Dias'  , color='green')
# #ax.plot(df_plot.index , df_plot[var_predicao + '_lag_7D'],label =  var_predicao  + ' Lag 7 Dias'  , color='green')
# #ax.plot(dados_x, df_plot[ var_predicao + ' Predito'] ,label = var_predicao  + ' Predito'  , color=COLOR_PRICE) 

# ax2.plot(dados_x, dados_y2 , color=COLOR_PRICE) 

# ax.set_title('Preço vs Gmv')
# ax.set_xlabel('Data') 

# leg = ax.legend()
# ax.legend(loc='upper left', frameon=False) 





# # %% 
# fig, ax = plt.subplots(figsize=(12,8))
# #ax2 = ax.twinx() 

# for color in range(1, df_plot_prices.shape[1] + 1):
#   prices = df_plot_prices.iloc[:,color-1:color].columns.to_list()[0]
#   if color == 1:
#     cor =  "#7900F1"
#   else:
#     if color ==2:
#       cor =  "#CD4C46"
#     elif color ==3:
#       cor =  "gray"
#     elif color ==4:
#       cor =  "black"
#     elif color ==5:
#       cor =  "yellow"
#     elif color ==6:
#       cor =  "purple"
#     elif color ==7:
#       cor =  "orange"
#     elif color ==8:
#       cor =  "red"
#     elif color ==9:
#       cor =  "blue"

#   ax.plot(df_plot_prices.index , df_plot_prices.iloc[:,color-1:color]  ,label =  prices  , color= cor) 
# #  ax.plot(df_plot_prices.index , df_plot_prices.iloc[:,0:2]  ,label =  'Prices'  , color=COLOR_TEMPERATURE) 
 

 
# ax.set_title('Prices')
# ax.set_xlabel('Data')
# leg = ax.legend()
# ax.legend(loc='upper left', frameon=False)
 



# %%  Plotly 

 


# # %% PLOTLY

# import plotly.graph_objects as go 
 
 

# min_price = df_plot_prices['Price Leite UHT Integral Elegê Caixa 1l - 7896079500151'].max() 
# max_price = df_plot_prices['Price Leite UHT Integral Elegê Caixa 1l - 7896079500151'].max()  


# min_gmv = 0 
# max_gmv = df_plot_prices['Gmv Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151'].max() + 1000 
# max_gmv
 

# fig = go.Figure(
#     data=go.Line(
#         x=dados_x,
#         y=df_plot_prices['Price Leite UHT Integral Elegê Caixa 1l - 7896079500151'],
#         name="Price",
#         marker=dict(color="#7900F1"),
#     )
# )

# fig.add_trace(
#     go.Scatter(
#         x=dados_x,
#         y=df_plot_prices['Gmv Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151'],
#         yaxis="y2",
#         name="Gmv",
#         marker=dict(color="crimson"),
#     )
# )

# fig.update_layout(
#     legend=dict(orientation="h"),
#     title = 'Preços x Gmv',

#     yaxis=dict(
#         title=dict(text="Price"),
#         side="left",
#         range=[min_price, max_price],
#     ),
#     yaxis2=dict(
#         title=dict(text="Gmv"),
#         side="right",
#         range=[0, max_gmv],
#         overlaying="y",
#         tickmode="sync",
#     ),
# )

# fig.show()

 



 


 

# %% Variáveis Modelo

# Variáveis Modelo
size = 15
size2 = 0
data_final = '2024-04-3'
region = 0
forecast_hora = False 
forecast_d0 = False
var_scaled = True               
var_log = False
  
categoria_quantity = []
categoria_gmv = []
categoria_positivacao = []
categoria_precos = []
categoria_precos_delta = []
categoria_ofertao = []
categoria_precos_delta_wind= []
categoria_precos_delta_lag = []

lista_categorias = list(map(lambda x: x.replace('Óleos, Azeites E Vinagres', 'Óleo'),lista_categorias ))
 


for categoria in lista_categorias:
  cate = categoria  

  for lista in df_modelo.columns.to_list():
#    print(lista)
    if lista.find(cate, 0)>=0: 
      if lista.find('Positivação',  0 )>=0: categoria_positivacao = categoria_positivacao  + [lista] 
      elif lista.find('Quantity',  0 )>=0: categoria_quantity = categoria_quantity   + [lista] 
      elif lista.find('Gmv',  0 )>=0: categoria_gmv = categoria_gmv  + [lista] 
      elif lista.find('Price',  0 )>=0: categoria_precos = categoria_precos  + [lista]
      elif lista.find('Delta Wind ',  0 )>=0: categoria_precos_delta_wind = categoria_precos_delta_wind  + [lista]
      elif lista.find('Delta Lag ',  0 )>=0: categoria_precos_delta_lag = categoria_precos_delta_lag  + [lista]
      elif lista.find('Ofertão',  0 )>=0: categoria_ofertao = categoria_ofertao  + [lista]


categoria_quantity = list(dict.fromkeys(categoria_quantity))
categoria_gmv = list(dict.fromkeys(categoria_gmv))
categoria_positivacao = list(dict.fromkeys(categoria_positivacao))
categoria_precos = list(dict.fromkeys(categoria_precos))
categoria_precos_delta_wind = list(dict.fromkeys(categoria_precos_delta_wind))
categoria_precos_delta_lag = list(dict.fromkeys(categoria_precos_delta_lag))
categoria_precos_delta = list(dict.fromkeys(categoria_precos_delta))
categoria_ofertao = list(dict.fromkeys(categoria_ofertao))
  

 
if forecast_hora == True:  

  df_modelo = df_modelo.merge(  pd.get_dummies(df_modelo['Hora']).astype(float).multiply(df_modelo['Gmv'], axis=0), how='left', left_index=True, right_index=True)    
  df_modelo['Gmv 8 Acum'] = df_modelo[0] + df_modelo[1] + df_modelo[2] + df_modelo[3] + df_modelo[4]  + df_modelo[5] + df_modelo[6] + df_modelo[7]    
  df_modelo['Gmv 10 Acum'] =  df_modelo['Gmv 8 Acum'] + + df_modelo[8] + df_modelo[9] 
  df_modelo['Gmv 12 Acum'] =  df_modelo['Gmv 10 Acum'] +  df_modelo[10] +  df_modelo[11] 
  df_modelo['Gmv 14 Acum'] =  df_modelo['Gmv 12 Acum'] +  df_modelo[12] +  df_modelo[13]  
  df_modelo['Gmv 16 Acum'] =  df_modelo['Gmv 14 Acum'] +  df_modelo[14] +  df_modelo[15]
  df_modelo['Gmv 18 Acum'] =  df_modelo['Gmv 16 Acum'] +  df_modelo[16] +  df_modelo[17]
  df_modelo['Gmv 20 Acum'] =  df_modelo['Gmv 18 Acum'] +  df_modelo[18] +  df_modelo[19]
  df_modelo['Gmv 22 Acum'] =  df_modelo['Gmv 20 Acum'] +  df_modelo[20] +  df_modelo[21]
  df_modelo['Gmv 24 Acum'] =  df_modelo['Gmv 22 Acum'] +  df_modelo[22] +  df_modelo[23] 
  df_modelo = df_modelo.drop(columns = pd.get_dummies(df_modelo['Hora']).columns.to_list() )
   
  var_lags = ['Gmv']  + ['Gmv 14 Acum' , 'Gmv 16 Acum'] 
  day_lag =  ["1D","5D", "14D","28D" ]
  var_wind = ['Gmv']  + ['Gmv 14 Acum' , 'Gmv 16 Acum']   
  day_wind = ["2D","14D","21D","28D","35D"]
 
  var_df = ['Gmv','Pedidos','Quantity'] + ['Gmv 14 Acum' , 'Gmv 16 Acum']
  var_df = ['Gmv'] + ['Gmv 14 Acum' , 'Gmv 16 Acum']

  lista_teste = ['week_sin','month_cos'] 
  lista_teste_barra = ['week_sin','month_cos'] + ['week', 'day_of_month_cos', 'weekend',  'month_sin'  ]  + ['Gmv 14 Acum_window_35D_mean'] 
  lista_teste = lista_teste_barra

  if forecast_d0: 
    var_drop =  [ 'Pedidos', 'Quantity'] + colnames_event  + col_pricing  +   ['Gmv 16 Acum']   
    var_drop =  colnames_event  + col_pricing  +   ['Gmv 16 Acum']   

  else:
    var_drop =  [ 'Pedidos', 'Quantity'] + colnames_event  + col_pricing  +  ['Gmv 14 Acum' , 'Gmv 16 Acum']   
    var_drop =  colnames_event  + col_pricing  +  ['Gmv 14 Acum' , 'Gmv 16 Acum']   
else: 

  # Lag e Janelas  
  day_lag =  [ "2D","5D","7D","28D","35D" ]
  day_wind =[ "5D","14D","21D","28D","35D"  ] 
    
  day_lag =  [ "2D","5D","7D","28D","35D" ]
  day_lag =  [ "7D","14D","21D"] 
  day_wind =[ "5D","14D","21D","28D","35D"  ] 
  day_wind = ["7D","28D"]   


#  var_df = ['Gmv']   
  var_drop =  ['Gmv']  + ['Pedidos']   + ['Trafego']  + ['Quantity']
  var_wind = ['Gmv']  + ['Pedidos']   + ['Trafego']
  var_lags = ['Gmv']  + ['Pedidos']   + ['Trafego']
  var_predicao = 'Gmv'    
  var_drop.remove(var_predicao)  
  var_bau = ['Gmv']  + ['Pedidos']  + ['Quantity'] 
  cols_trafego = ['Trafego','Trafego_Search_Products', 'Trafego_Add_To_Cart','Trafego_Checkout']
  #var_drop.remove('Trafego') # + ['Trafego'] # +  ['Gmv']  + ['Pedidos'] + cols_trafego

 
# Lag e Janelas  
day_lag = [ "1D","2D","5D","6D","7D","12D","14D","21D","28D","35D","84D" ]
day_wind = [ "1D","2D","5D","7D","12D","14D","21D","28D","35D","84D"  ] 


day_lag = [ "1D", "5D","6D","7D","8D","12D","14D","21D","28D" ]
day_wind = [ "1D","2D","5D","7D","12D","14D","21D","28D","35D","84D"  ] 
 
 
#prod_lista # Price 
#mean_lista #Lag- 
#wind_lista #Wind 
#delta_wind #Delta Wind 
#delta_lag_mean #Delta  
#delta_lag #Delta Lag
#delta_conc #Delta Conc   

var_drop = categoria_gmv.copy()  #+ ['Trafego']
var_wind = ['Gmv Categoria_' + categoria] # + ['Trafego']
var_lags = ['Gmv Categoria_' + categoria] #+ ['Trafego']  
var_predicao = 'Gmv'
var_predicao = 'Gmv Categoria_' + categoria  
var_drop.remove(var_predicao) 
 
 
cols_price_clubbi = [i for i in cols_price_clubbi if i.find('1-4 Cxs' ,0)>0] 
delta_conc = [i for i in delta_conc if i.find('Torre_Cia' ,0)>0 or i.find('Mundial' ,0)>0] 
 
delta_lag_mean = [i for i in delta_lag_mean if i.find('1-4 Cxs' ,0)>0] 
 
df_modelo = df_modelo[['Trafego'] + categoria_gmv + cols_price_clubbi  +   delta_lag_mean + colnames_event + ofertao_columns]   
df_modelo = df_modelo[  categoria_gmv + cols_price_clubbi  +   delta_lag_mean + colnames_event + ofertao_columns]   


#cols_price_concorrencia
  
date_cols = [
    'Seg','Ter', 'Qua','Qui','Sex', 'Sáb', 'Dom',
    'month',
    'week',
    'day_of_week',
    'day_of_month',
    #'weekend',
    #'month_sin',
    'month_cos',
    #'week_sin',
    'week_cos',
    #'day_of_week_sin',
    'day_of_week_cos',
    #'day_of_month_sin',
    #'day_of_month_cos'
] 
outras_cols = [
#    'Ofertão Biscoitos',
    'week_sin','month_cos','month_sin','weekend',
    'Gmv_Shift', 'Quantity_Shift','Pedidos_Shift','Gmv_inicial'
#    'Ofertão Óleos, Azeites E Vinagres',
#    'Ofertão Chocolates'
] 

if var_scaled==True: 
      
    trafego_cols_scaled =  [
    #  'Trafego_lag_1D Scaled',
    #  'Trafego_lag_2D Scaled',
    #  'Trafego_lag_5D Scaled',
       'Trafego_lag_7D Scaled',
    #  'Trafego_lag_14D Scaled',
    #  'Trafego_lag_21D Scaled',
      'Trafego_lag_28D Scaled',
      'Trafego_lag_35D Scaled',
    #  'Trafego_window_1D_mean Scaled',
    #  'Trafego_window_2D_mean Scaled',
    #  'Trafego_window_5D_mean Scaled',
    #  'Trafego_window_7D_mean Scaled',
      'Trafego_window_14D_mean Scaled',
     # 'Trafego_window_21D_mean Scaled',
     # 'Trafego_window_28D_mean Scaled',
     # 'Trafego_window_35D_mean Scaled',
    #  'Trafego_window_84D_mean Scaled',
     # 'Trafego_L2W Scaled',
      'Trafego_L3W Scaled',
      'Trafego_L4W Scaled',
     # 'Trafego_lag_84D Scaled',
      ]
    gmv_cols_scaled = [        
          #'Gmv_lag_1D Scaled',
          'Gmv_lag_2D Scaled',
          'Gmv_lag_5D Scaled',
          'Gmv_lag_7D Scaled',
          #'Gmv_lag_14D Scaled',
          #'Gmv_lag_21D Scaled',
          'Gmv_lag_28D Scaled',
          'Gmv_lag_35D Scaled',
          #'Gmv_window_1D_mean Scaled',
          #'Gmv_window_2D_mean Scaled',
          'Gmv_window_5D_mean Scaled',
          #'Gmv_window_7D_mean Scaled',
          'Gmv_window_14D_mean Scaled',
    #      'Gmv_window_21D_mean Scaled',
          'Gmv_window_28D_mean Scaled',
          'Gmv_window_35D_mean Scaled',
        # 'Gmv_window_84D_mean Scaled',
          'Gmv_L2W Scaled',
          'Gmv_L3W Scaled',
          'Gmv_L4W Scaled',
        # 'Gmv_lag_84D Scaled',
    ]
    pedidos_cols_scaled = [
        #'Pedidos_lag_1D Scaled',
        'Pedidos_lag_2D Scaled',
      #  'Pedidos_lag_5D Scaled',
    #   'Pedidos_lag_7D Scaled',
        #'Pedidos_lag_14D Scaled',
        #'Pedidos_lag_21D Scaled', 
        'Pedidos_lag_28D Scaled', 
        'Pedidos_lag_35D Scaled',
        #'Pedidos_window_1D_mean Scaled',
        #'Pedidos_window_2D_mean Scaled',
        #'Pedidos_window_5D_mean Scaled',
        #'Pedidos_window_7D_mean Scaled',
        'Pedidos_window_14D_mean Scaled',
    #    'Pedidos_window_21D_mean Scaled',
        'Pedidos_window_28D_mean Scaled',
      # 'Pedidos_window_35D_mean Scaled',
        #'Pedidos_window_84D_mean Scaled',
        #'Pedidos_L2W Scaled',
        #'Pedidos_L3W Scaled',
      # 'Pedidos_L4W Scaled',
      # 'Pedidos_lag_84D Scaled',
    ]
    cols_ofertao =   [
    #'Ofertão',
    'Ofertão Óleos, Azeites E Vinagres',
    #'Ofertão Leite',
    # 'Ofertão Leites E Derivados',
    # 'Ofertão Açúcar E Adoçante',
    # 'Ofertão Biscoitos',
    #'Ofertão Energéticos',
    # 'Ofertão Cervejas',
    #'Ofertão Cervejas Premium',
    # 'Ofertão Chocolates',
    #'Ofertão Arroz E Feijão',
    #'Ofertão Cafés, Chás E Achocolatados'
    ]
    cols_categoria = [
        
      'Gmv Categoria_Leite_lag_5D Scaled',
      'Gmv Categoria_Leite_lag_7D Scaled',
      'Gmv Categoria_Leite_lag_28D Scaled',
      'Gmv Categoria_Leite_lag_35D Scaled',
      'Gmv Categoria_Leite_window_14D_mean Scaled', 
      'Gmv Categoria_Leite_window_21D_mean Scaled', 
    # 'Gmv Categoria_Leite_window_28D_mean Scaled', 
    #  'Gmv Categoria_Leite_window_35D_mean Scaled',  

      
      'Gmv Categoria_Açúcar e Adoçante_lag_5D Scaled',
    #  'Gmv Categoria_Açúcar e Adoçante_lag_7D Scaled',
      'Gmv Categoria_Açúcar e Adoçante_lag_28D Scaled',
      'Gmv Categoria_Açúcar e Adoçante_lag_35D Scaled',
      'Gmv Categoria_Açúcar e Adoçante_window_14D_mean Scaled', 
    #  'Gmv Categoria_Açúcar e Adoçante_window_21D_mean Scaled', 
    # 'Gmv Categoria_Açúcar e Adoçante_window_28D_mean Scaled', 
    # 'Gmv Categoria_Açúcar e Adoçante_window_35D_mean Scaled',  

    

    # 'Gmv Categoria_Óleos, Azeites E Vinagres_lag_5D Scaled',
      'Gmv Categoria_Óleos, Azeites E Vinagres_lag_7D Scaled',
      #'Gmv Categoria_Óleos, Azeites E Vinagres_lag_28D Scaled',
      #'Gmv Categoria_Óleos, Azeites E Vinagres_lag_35D Scaled',
      #'Gmv Categoria_Óleos, Azeites E Vinagres_window_14D_mean Scaled', 
      'Gmv Categoria_Óleos, Azeites E Vinagres_window_21D_mean Scaled', 
      #'Gmv Categoria_Óleos, Azeites E Vinagres_window_28D_mean Scaled', 
    #  'Gmv Categoria_Óleos, Azeites E Vinagres_window_35D_mean Scaled', 
    
    # AZEITE
      'Gmv Categoria_Cervejas_lag_5D Scaled',
    #   'Gmv Categoria_Cervejas_lag_7D Scaled',
    #   'Gmv Categoria_Cervejas_lag_28D Scaled',
    #   'Gmv Categoria_Cervejas_lag_35D Scaled',
    #  'Gmv Categoria_Cervejas_window_14D_mean Scaled', 
    #   'Gmv Categoria_Cervejas_window_21D_mean Scaled', 
      'Gmv Categoria_Cervejas_window_28D_mean Scaled', 
    #   'Gmv Categoria_Cervejas_window_35D_mean Scaled', 
      
      

    # NOVOS 


      # 'Gmv Categoria_Chocolates_lag_5D Scaled',
      #'Gmv Categoria_Chocolates_lag_7D Scaled',
      
      # 'Gmv Categoria_Chocolates_lag_28D Scaled',
      # 'Gmv Categoria_Chocolates_lag_35D Scaled',
      
    #'Gmv Categoria_Chocolates_window_7D_mean Scaled',
    # 'Gmv Categoria_Chocolates_window_14D_mean Scaled', 
      # 'Gmv Categoria_Chocolates_window_21D_mean Scaled', 
      'Gmv Categoria_Chocolates_window_28D_mean Scaled', 
      # 'Gmv Categoria_Chocolates_window_35D_mean Scaled', 

    

      # 'Gmv Categoria_Biscoitos_lag_5D Scaled',
    #   'Gmv Categoria_Biscoitos_lag_7D Scaled',
      #'Gmv Categoria_Biscoitos_lag_28D Scaled',
      # 'Gmv Categoria_Biscoitos_lag_35D Scaled',
      # 'Gmv Categoria_Biscoitos_window_14D_mean Scaled', 
      # 'Gmv Categoria_Biscoitos_window_21D_mean Scaled', 
        'Gmv Categoria_Biscoitos_window_28D_mean Scaled', 
      # 'Gmv Categoria_Biscoitos_window_35D_mean Scaled', 



      # 'Gmv Categoria_Derivados de Leite_lag_5D Scaled',
      # 'Gmv Categoria_Derivados de Leite_lag_7D Scaled',
      # 'Gmv Categoria_Derivados de Leite_lag_28D Scaled',
      # 'Gmv Categoria_Derivados de Leite_lag_35D Scaled',
      # 'Gmv Categoria_Derivados de Leite_window_14D_mean Scaled', 
      # 'Gmv Categoria_Derivados de Leite_window_21D_mean Scaled', 
      'Gmv Categoria_Derivados de Leite_window_28D_mean Scaled', 
      # 'Gmv Categoria_Derivados de Leite_window_35D_mean Scaled', 
    


      # 'Gmv Categoria_Arroz e Feijão_lag_5D Scaled',
      'Gmv Categoria_Arroz e Feijão_lag_7D Scaled',
    #  'Gmv Categoria_Arroz e Feijão_lag_14D Scaled',
    #   'Gmv Categoria_Arroz e Feijão_lag_28D Scaled',
      # 'Gmv Categoria_Arroz e Feijão_lag_35D Scaled',
      # 'Gmv Categoria_Arroz e Feijão_window_14D_mean Scaled', 
      'Gmv Categoria_Arroz e Feijão_window_21D_mean Scaled', 
    #  'Gmv Categoria_Arroz e Feijão_window_28D_mean Scaled', 
      # 'Gmv Categoria_Arroz e Feijão_window_35D_mean Scaled', 
      
    
    ]
    date_cols_check = [
        'Seg','Ter', 'Qua','Qui','Sex', 'Sáb', 'Dom',
        'month',
        'week',
        'day_of_week',
        'day_of_month',
        #'weekend',
        #'month_sin',
        'month_cos',
        #'week_sin',
        'week_cos',
        #'day_of_week_sin',
        'day_of_week_cos',
        #'day_of_month_sin',
        #'day_of_month_cos'
    ] 
    cols_categoria_check = [
        
      
      'Gmv Categoria_Óleos, Azeites E Vinagres_lag_2D Scaled', 
      'Gmv Categoria_Óleos, Azeites E Vinagres_lag_5D Scaled',
      'Gmv Categoria_Óleos, Azeites E Vinagres_lag_7D Scaled',
      'Gmv Categoria_Óleos, Azeites E Vinagres_lag_12D Scaled',
      'Gmv Categoria_Óleos, Azeites E Vinagres_lag_21D Scaled',
      'Gmv Categoria_Óleos, Azeites E Vinagres_lag_28D Scaled',
      'Gmv Categoria_Óleos, Azeites E Vinagres_lag_35D Scaled',
    #  'Gmv Categoria_Óleos, Azeites E Vinagres_window_14D_mean Scaled', 
      'Gmv Categoria_Óleos, Azeites E Vinagres_window_21D_mean Scaled', 
      'Gmv Categoria_Óleos, Azeites E Vinagres_window_28D_mean Scaled', 
    # 'Gmv Categoria_Óleos, Azeites E Vinagres_window_35D_mean Scaled', 
      
      
    
    ]

 
 
    tail_col = [ 

    'Eventos',  
    'Dom',
 'Seg',
 'Ofertão Leite Leite UHT Integral Ninho Forti+ Caixa com Tampa 1l - 7898215157403',
 'Sex',
 'Qua',
 'Ofertão Leite Leite UHT Integral Italac Com Tampa 1l - 7898080640611',
 'month_cos',
 'Ofertão',
 'week_sin',
 'Price BAU 1-4 Cxs - Leite Uht Integral Elegê Caixa 1L - 7896079500151 Scaled',
 'month_sin',
 'weekend',
 'Price BAU 1-4 Cxs - Leite Uht Integral Ninho Forti+ Caixa Com Tampa 1L - 7898215157403 Scaled',
 'Price BAU 1-4 Cxs - Leite Uht Integral Quatá 1L - 7896183202187 Scaled',
 'Delta Lag- Mean 7/14/21 Price BAU 1-4 Cxs - Leite Uht Integral Elegê Caixa 1L - 7896079500151 Scaled',
 'Delta Lag- Mean 7/14/21/28 Price BAU 1-4 Cxs - Leite Uht Integral Elegê Caixa 1L - 7896079500151 Scaled',
 'Delta Lag- Mean 7/14/21/28 Price BAU 1-4 Cxs - Leite Uht Integral Quatá 1L - 7896183202187 Scaled',
 'Delta Lag- Mean 7/14 Price BAU 1-4 Cxs - Leite Uht Integral Piracanjuba Caixa Com Tampa 1L - 7898215151708 Scaled',
 'Delta Lag- Mean 7/14/21 Price BAU 1-4 Cxs - Leite Uht Integral Piracanjuba Caixa Com Tampa 1L - 7898215151708 Scaled',
 'Delta Lag- Mean 7/14/21/28 Price BAU 1-4 Cxs - Leite Uht Integral Piracanjuba Caixa Com Tampa 1L - 7898215151708 Scaled',
 'Delta Lag- Mean 7/14/21 Price BAU 1-4 Cxs - Leite Uht Integral Ninho Forti+ Caixa Com Tampa 1L - 7898215157403 Scaled',
 'Delta Lag- Mean 7/14/21/28 Price BAU 1-4 Cxs - Leite Uht Integral Ninho Forti+ Caixa Com Tampa 1L - 7898215157403 Scaled',
 'Gmv Categoria_Leite_window_2D_mean Scaled',
 'Gmv Categoria_Leite_window_5D_mean Scaled',
 'Price BAU 1-4 Cxs - Leite Uht Integral Italac Com Tampa 1L - 7898080640611 Scaled',

 'Ofertão Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151',
 'Gmv Categoria_Leite_lag_21D Scaled',
 'Gmv Categoria_Leite_window_7D_mean Scaled',
 'Delta Lag- Mean 7/14 Price BAU 1-4 Cxs - Leite Uht Integral Italac Com Tampa 1L - 7898080640611 Scaled',
 'Gmv Categoria_Leite_window_1D_mean Scaled',
 'Gmv Categoria_Leite_lag_8D Scaled', 

 'Ofertão Leite Leite UHT Integral Quatá 1L - 7896183202187',
 'Delta Lag- Mean 7/14 Price BAU 1-4 Cxs - Leite Uht Integral Ninho Forti+ Caixa Com Tampa 1L - 7898215157403 Scaled',
 'day_of_month_sin',
 'day_of_week_cos',
 'week',
 'day_of_month',
 'month',
 'Delta Lag- Mean 7/14/21 Price BAU 1-4 Cxs - Leite Uht Integral Quatá 1L - 7896183202187 Scaled',
 'Ofertão Leite Leite UHT Integral Piracanjuba Caixa com Tampa 1L - 7898215151708',
 'Delta Lag- Mean 7/14/21/28 Price BAU 1-4 Cxs - Leite Uht Integral Italac Com Tampa 1L - 7898080640611 Scaled',
 'Qui',
 'day_of_month_cos',
 'day_of_week_sin',
 'week_cos',
 'Gmv Categoria_Leite_lag_12D Scaled',
 
 #'Gmv Categoria_Leite_window_28D_mean Scaled',
 'Gmv Categoria_Leite_window_12D_mean Scaled',
 'Gmv Categoria_Leite_window_35D_mean Scaled',
 

]


    drop_var_final =  [var_predicao] + date_cols_check + ofertao_columns 
 
else:
    trafego_cols = [       
      'Trafego_lag_2D',
      'Trafego_lag_5D',
      'Trafego_lag_7D', 
      'Trafego_lag_28D',
      'Trafego_lag_35D', 
      'Trafego_window_5D_mean', 
      'Trafego_window_14D_mean',
      'Trafego_window_21D_mean',
      'Trafego_window_28D_mean',
      'Trafego_window_35D_mean',
      
#      'Trafego_L2W',
#      'Trafego_L3W',
#      'Trafego_L4W',
#      'Trafego_lag_1D',
  ]
    pedidos_cols = [
    #'Pedidos_L2W',
    #'Pedidos_L3W',
    #'Pedidos_L4W',
    #'Pedidos_lag_1D',
    #'Pedidos_lag_2D',
    'Pedidos_lag_5D',
    'Pedidos_lag_7D',
    #'Pedidos_lag_14D',
    #'Pedidos_lag_21D',
    'Pedidos_lag_28D',
    'Pedidos_lag_35D',
    #'Pedidos_window_1D_mean',
    #'Pedidos_window_2D_mean',
    'Pedidos_window_5D_mean',
    #'Pedidos_window_7D_mean',
    'Pedidos_window_14D_mean',
    'Pedidos_window_21D_mean',
    'Pedidos_window_28D_mean',
    'Pedidos_window_35D_mean'
]
    gmv_cols =  [
        'Gmv',
        #'Gmv_L2W',
        #'Gmv_L3W',
        #'Gmv_L4W',
        #'Gmv_lag_1D',
        'Gmv_lag_2D',
        'Gmv_lag_5D',
        'Gmv_lag_7D',
        #'Gmv_lag_14D',
        #'Gmv_lag_21D',
        'Gmv_lag_28D',
        'Gmv_lag_35D',
        #'Gmv_window_1D_mean',
        #'Gmv_window_2D_mean',
        'Gmv_window_5D_mean',
        #'Gmv_window_7D_mean',
        'Gmv_window_14D_mean',
        'Gmv_window_21D_mean',
        'Gmv_window_28D_mean',
        'Gmv_window_35D_mean'
]
    drop_var_final =  [var_predicao] + gmv_cols + pedidos_cols   + trafego_cols + date_cols  



# ACF e PACF
# Serie Não estacionária, n serve para ver lags
 
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 

plot_acf(df_modelo[var_predicao].dropna(),lags=50)
plt.show()
   
plot_pacf(df_modelo[var_predicao].dropna(),lags=50)
plt.show() 


# %% Df Modelo Transform
# Df Modelo Transform


dtf = DatetimeFeatures(variables = "index",features_to_extract = ["month","week","day_of_week","day_of_month","weekend"]) 
cyclicf = CyclicalFeatures(variables = ["month","week","day_of_week", "day_of_month"],drop_original=False,)
lagf = LagFeatures(variables = var_lags ,freq = day_lag  ,missing_values = "ignore",) 
winf = WindowFeatures(variables = var_wind ,  window =  day_wind ,freq = "1D", missing_values = "ignore",) 
#winf = WindowFeatures(variables = var,window = ["1H","2H","4H","6H","8H","10H","12H" ,"24H","168H","1000H"],freq = "1H", missing_values = "ignore",)
imputer = DropMissingData()
drop_ts = DropFeatures(features_to_drop =  var_drop)
pipe_date = Pipeline([
  ("datetime_features",dtf), 
  ("Periodic",cyclicf),
]) 
pipe = Pipeline([ 
  ("lagf",lagf), 
  ("winf",winf),
  ("dropna",imputer),
  ("drop_ts",drop_ts),
]) 

pre_pipe = df_modelo.columns.to_list()
pre_pipe
  
df_modelo = pipe_date.fit_transform(df_modelo)     
date_features = df_modelo.drop(columns=pre_pipe).columns.to_list() + ['Seg','Ter','Qua','Qui','Sex','Sáb','Dom']
df_modelo = pipe.fit_transform(df_modelo)      
 
 
# df_modelo['Pedidos_L2W'] = (df_modelo['Pedidos_lag_7D'] + df_modelo['Pedidos_lag_14D'] )/2
# df_modelo['Pedidos_L3W'] = (df_modelo['Pedidos_lag_7D'] + df_modelo['Pedidos_lag_14D'] + df_modelo['Pedidos_lag_21D'] )/3
# df_modelo['Pedidos_L4W'] = (df_modelo['Pedidos_lag_7D'] + df_modelo['Pedidos_lag_14D'] + df_modelo['Pedidos_lag_21D'] + df_modelo['Pedidos_lag_28D'])/4

# df_modelo['Gmv_L2W'] = (df_modelo['Gmv_lag_7D'] + df_modelo['Gmv_lag_14D'] )/2
# df_modelo['Gmv_L3W'] = (df_modelo['Gmv_lag_7D'] + df_modelo['Gmv_lag_14D'] + df_modelo['Gmv_lag_21D'] )/3
# df_modelo['Gmv_L4W'] = (df_modelo['Gmv_lag_7D'] + df_modelo['Gmv_lag_14D'] + df_modelo['Gmv_lag_21D'] + df_modelo['Gmv_lag_28D'])/4

# df_modelo['Trafego_L2W'] = (df_modelo['Trafego_lag_7D'] + df_modelo['Trafego_lag_14D'] )/2
# df_modelo['Trafego_L3W'] = (df_modelo['Trafego_lag_7D'] + df_modelo['Trafego_lag_14D'] + df_modelo['Trafego_lag_21D'] )/3
# df_modelo['Trafego_L4W'] = (df_modelo['Trafego_lag_7D'] + df_modelo['Trafego_lag_14D'] + df_modelo['Trafego_lag_21D'] + df_modelo['Trafego_lag_28D'])/4

df_modelo['Seg'] = np.where((df_modelo['day_of_week'] == 0) , 1 , 0 )
df_modelo['Ter'] = np.where((df_modelo['day_of_week'] == 1) , 1 , 0 )
df_modelo['Qua'] = np.where((df_modelo['day_of_week'] == 2) , 1 , 0 )
df_modelo['Qui'] = np.where((df_modelo['day_of_week'] == 3) , 1 , 0 )
df_modelo['Sex'] = np.where((df_modelo['day_of_week'] == 4) , 1 , 0 )
df_modelo['Sáb'] = np.where((df_modelo['day_of_week'] == 5) , 1 , 0 )
df_modelo['Dom'] = np.where((df_modelo['day_of_week'] == 6) , 1 , 0 )

 

df_modelo_transform = df_modelo.copy() 

 
trafego_columns = df_modelo.filter(regex='Trafego').columns.to_list()
gmv_columns = df_modelo.filter(regex='Gmv').columns.to_list()
pedidos_columns = df_modelo.filter(regex='Pedidos').columns.to_list() 

# print('Features Pre Scaled')
# print(trafego_columns)
# print(gmv_columns)
# print(pedidos_columns)
 
 
 
if forecast_d0 == True:  
  df_modelo['Var Wind 14 - 14 Mean'] =  (df_modelo['Gmv 14 Acum'].astype('float')  - df_modelo['Gmv 14 Acum_window_14D_mean'].astype('float') )
  df_modelo['Var Lag 14'] =  (df_modelo['Gmv 14 Acum'].astype('float')  - df_modelo['Gmv 14 Acum_lag_14D'].astype('float') )
  df_modelo = df_modelo.drop(columns=['Gmv 14 Acum_window_14D_mean','Gmv 14 Acum_lag_14D'])
 
if var_scaled == True: 

  scaler = preprocessing.StandardScaler()
  arr_scaled = scaler.fit_transform(df_modelo.drop(columns = date_features + [var_predicao]  + colnames_event  + ofertao_columns ).reset_index(drop = True))
  df_scaled = pd.DataFrame(arr_scaled, columns = df_modelo.drop(columns = date_features + [var_predicao] + colnames_event  + ofertao_columns ).columns)
  df_scaled.columns=[  str(df_scaled.columns[k-1]) + " Scaled" for k in range(1, df_scaled.shape[1] + 1)]
  df_modelo = df_modelo.reset_index(drop = False).merge(df_scaled, left_index = True, right_index=True,how = "left").set_index('Data')
  df_modelo = df_modelo[[var_predicao] +  df_scaled.columns.to_list() +  date_features + colnames_event + ofertao_columns ]

elif var_log == True:

  df_modelo[df_modelo.drop(columns=['Gmv'] + date_features ).columns.to_list()] = np.log(df_modelo.drop(columns=['Gmv'] + date_features))
  df_modelo.replace([np.inf, -np.inf], 0, inplace=True)

 

trafego_columns = df_modelo.filter(regex='Trafego').columns.to_list()
gmv_columns = df_modelo.filter(regex='Gmv').columns.to_list()
positivacao_columns = df_modelo.filter(regex='Positivação').columns.to_list()
qtd_columns = df_modelo.filter(regex='Quantity').columns.to_list()
pedidos_columns = df_modelo.filter(regex='Pedidos').columns.to_list() 
delta_wind_columns = df_modelo.filter(regex='Delta Wind').columns.to_list() 
delta_lag = df_modelo.filter(regex='Delta Lag-').columns.to_list() 
price_columns = df_modelo.filter(regex='Price').columns.to_list() 
  

price_columns = [i for i in price_columns if i not in delta_wind_columns] 
price_columns = [i for i in price_columns if i not in delta_lag] 
price_columns
 
#df_modelo = df_modelo[drop_var_final + delta_wind_columns + positivacao_columns + qtd_columns + gmv_columns ] 

#df_modelo = df_modelo[drop_var_final +   gmv_columns + price_columns + trafego_columns] 

#df_modelo = df_modelo.drop(columns = tail_col)
#print('Variavéis Modelo')
#print(df_modelo.columns.to_list()) 



#df_modelo = df_modelo.drop(columns = df_modelo.filter(regex='Trafego').columns.to_list())
 
df_modelo


# %% Modelo
#  Modelo 

 
data_ref = df_modelo.reset_index(drop=False)['Data'].dt.date.max() 
data_corte = pd.Timestamp(data_ref)  - pd.offsets.Day(size+size2) 


# Treino e Teste 
 
x_train = df_modelo[df_modelo.index <= data_corte ].drop(columns = [var_predicao])
x_train = x_train.dropna()
 
x_train

 
x_test = df_modelo[df_modelo.index > data_corte].drop(columns = [var_predicao])
y_train = df_modelo[df_modelo.index <=  data_corte][[var_predicao]]
y_test = df_modelo[df_modelo.index > data_corte ][[var_predicao]]
y_test = y_test.replace(np.nan, 0)
x_test = x_test.dropna( )
#x_test['Eventos'] = x_test['Eventos'].replace(1,0)
 
y_train = y_train.loc[x_train.index] 

y_train = y_train.iloc[:,0:1]

y_test = y_test.iloc[:,0:1]
 
# Modelo 


lasso = Lasso( alpha = 100, random_state = 0 )
lasso.fit(x_train, y_train)
preds2 = lasso.predict(x_test)
  
 
 
# Output e Erro 

coef = pd.DataFrame(pd.Series( lasso.coef_, index = x_train.columns), columns=['Coef']) 
coef['Coef Abs'] = np.abs(coef['Coef'])
df_predito = pd.DataFrame(preds2,columns=[var_predicao + ' Predito']) 

 

y_test = y_test.reset_index(drop= False)
y_test = y_test.merge(  df_predito, how='left', left_index=True, right_index=True)  
y_test = y_test.set_index('Data')
df_erro = y_test[:len(y_test) +1 - size2]
df_erro = df_erro.dropna()
df_plot = df_erro.copy()
df_plot
 
 
df_erro = df_erro.iloc[:df_erro.shape[0]-1,:]
 
 
rmse = sqrt(mean_squared_error(df_erro[var_predicao], df_erro[var_predicao + ' Predito']))
mape =  mean_absolute_percentage_error(df_erro[var_predicao], df_erro[var_predicao + ' Predito'])
 
   
print('Features')
print(x_test.columns.to_list())
# Plots e Prints 

# plt.scatter(y_test['Gmv Predito'][:len(y_test)-1],y_test['Gmv'][:len(y_test)-1] )
# plt.xlabel("gmv pred")
# plt.ylabel("gmv")
# plt.title("Forecast: train set") 


print('Region ')
#print(region)
print('Data Corte ' + str(data_corte)) 

print(rmse)
print("MAPE: {:.2f}%".format(mape*100)) 
coef = coef.sort_values(by=['Coef Abs'], ascending= False)

coef 
 
# %% Tail Coef 

 
tail_columns = coef.tail(30).index.to_list()

coef.tail(30)
tail_columns

# %% Df Erro 
   
#df_plot = df_modelo_transform[ofertao_columns  + [ 'Evento_Carnaval', var_predicao , var_predicao + '_lag_7D',var_predicao + '_window_28D_mean']][df_modelo_transform.index > pd.Timestamp('2023-11-01')].merge(df_plot[var_predicao + ' Predito'], left_index = True, right_index=True,how = "left")

df_plot = df_modelo_transform[df_modelo_transform.index > pd.Timestamp('2023-11-01')].merge(df_plot[var_predicao + ' Predito'], left_index = True, right_index=True,how = "left")
df_plot = df_plot[[  var_predicao,  var_predicao + ' Predito'] ]
#df_plot.to_excel('Teste.xlsx')

df_plot.tail(60)
 
# %% Plot



COLOR_TEMPERATURE = "#7900F1"
COLOR_PRICE = "#CD4C46"

 
fig, ax = plt.subplots(figsize=(12,5))
#ax2 = ax.twinx() 
ax.plot(df_plot.index , df_plot[var_predicao] ,label =  var_predicao  , color=COLOR_TEMPERATURE) 
#ax.plot(df_plot.index , df_plot[var_predicao + '_window_28D_mean'],label =  var_predicao  + ' Média 28 Dias'  , color='green')
#ax.plot(df_plot.index , df_plot[var_predicao + '_lag_7D'],label =  var_predicao  + ' Lag 7 Dias'  , color='green')
ax.plot(df_plot.index, df_plot[ var_predicao + ' Predito'] ,label = var_predicao  + ' Predito'  , color=COLOR_PRICE) 

#ax2.plot(df_plot.index, df_plot[ var_predicao + ' Predito'], color=COLOR_PRICE) 

ax.set_title( var_predicao + 'vs ' + var_predicao +  ' Predito')
ax.set_xlabel('Data')


#ax.legend([var_predicao], loc='upper left')
#ax2.legend([var_predicao + ' Predito'], loc='upper right')


leg = ax.legend()
ax.legend(loc='upper left', frameon=False)
 
#df_erro['Delta'] = df_erro['Gmv']  - df_erro['Gmv Predito']
#df_erro.sort_values('Delta',ascending = False )
#df_erro.to_excel('C:/Users/leona/Área de Trabalho/Prophet/Clientes/Erro_Modelo.xlsx') 
 
plt.show()
 
  
 
# %% Correlação Variavéis 

# Outras Metricas 
# Rodar 1-4 Cxs - Mercados   - Check
# AOV - Check
# Delta Preço Concorrencia - Check
# Qtd Skus (Top 80)/Top 80 - Check
# Métricas Trafego  - Check 
 
 
# Metrica de % Risco e % Ruptura



df_correl = df_correl_guarda.copy()
#df_correl = df_correl[df_correl['Ofertão Óleos, Azeites E Vinagres']>0]
 
df_correl[cols_price_clubbi]
 

categoria = df_correl.columns[df_correl.columns.str.startswith('Gmv Categoria_')].tolist()
categoria = categoria[0][ len('Gmv Categoria_'):]
categoria = 'Categoria_' + categoria 

df_correl['%Top_Produtos'] = df_correl['Top Produtos ' + categoria]/df_correl['Top Produtos Total ' + categoria]
df_correl['%Top_Produtos'] = df_correl['%Top_Produtos'].replace(np.nan,0)
df_correl['%Share_Gmv_Categoria'] = df_correl['Gmv ' + categoria]/df_correl['Gmv']
df_correl['%Positivação_Categoria'] = df_correl['Positivação ' + categoria]/df_correl['Pedidos']
df_correl['%Conversao_Categoria'] = df_correl['Positivação ' + categoria]/df_correl['Trafego']
df_correl['Aov_Categoria'] = np.where((df_correl['Gmv ' + categoria] > 0) ,   df_correl['Gmv ' + categoria]/df_correl['Positivação ' + categoria]   , 0 )
df_correl['Aov_Categoria'] = df_correl['Aov_Categoria'].replace(np.nan,0)
#df_correl = df_correl[df_correl['Gmv '+ categoria]>0]



#df_teste = df_teste[df_teste['Price']>df_teste["Price"].quantile(0.01)]

#df_correl
 
lista_bau = ['%Share_Gmv_Categoria','%Positivação_Categoria','%Conversao_Categoria','Aov_Categoria','%Top_Produtos']
#lista_prices =  df_correl[col_delta_prices + cols_price_clubbi].columns.to_list()

lista_prices =  df_correl[cols_price_clubbi + cols_price_concorrencia  + delta_lag_mean + delta_conc].columns.to_list()
lista_prices =  df_correl[cols_price_clubbi ].columns.to_list()
 
#df_correl_guarda.columns[df_correl_guarda.columns.str.startswith('Delta Lag-')].tolist()   



# %% Heatmap 



df_outliers_leite = pd.read_excel('C:/Users/leona/Área de Trabalho/Prophet/Outlier Data.xlsx')  
df_outliers_leite = df_outliers_leite.set_index('Data')
df_outliers_leite
 
df_correl_teste = df_correl.copy() 
df_correl_teste = df_correl_teste .merge(  df_outliers_leite, how='left', left_index=True, right_index=True)  
df_correl_teste = df_correl_teste[df_correl_teste['Outlier']==0]
df_correl_teste
 
#lista_correl = ['Gmv ' + categoria] + df_regressao.index.to_list() 
#lista_correl.remove('const')
lista_correl = cols_gmv_categoria + ['Price Leite UHT Integral Elegê Caixa 1l - 7896079500151'] + lista_bau + delta_conc  + delta_lag_mean +  ofertao_columns  

lista_correl.remove('Gmv Categoria_Leite')

#df_teste = df_teste[df_teste['Price']>df_teste["Price"].quantile(0.01)]
df_correl = df_correl[lista_correl]
df_correl = df_correl[df_correl['Gmv Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151']> df_correl['Gmv Categoria Leite Leite UHT Integral Elegê Caixa 1l - 7896079500151'].quantile(0.2)]
df_correl
 

#fig, ax = plt.subplots(figsize=(30,7))        
correlation_matrix = df_correl.reset_index(drop=True).corr()
correlation_matrix = df_correl[lista_correl].reset_index(drop=True).corr()
 
#sns.heatmap(correlation_matrix , annot=True, linewidths=.5, ax=ax) 
#plt.title("Correlation Categoria")
#plt.ylabel("Features") 
#plt.show()
 

correlation_matrix
# %% Heatmap Categoria


fig, ax = plt.subplots(figsize=(2,60))        
correlation_category = correlation_matrix.iloc[:,0:1].sort_values(correlation_matrix.iloc[:,0:1].columns.to_list()[0],ascending = False )
sns.heatmap(correlation_category , annot=True, linewidths=.5, ax=ax) 
plt.title("Correlation Categoria")
plt.ylabel("Features") 
plt.show()



# %% Modelo Regerssão Linear
 
lista_correl =  lista_bau + lista_prices 
delta_lag_fim = [

'Delta Lag- Mean 7/14/21/28 Price Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244',
'Delta Lag- Mean 7/14/21/28 Price Óleo de Soja Soya Garrafa 900ml - 7891107101621',
'Delta Lag- Mean 7/14/21/28 Price Óleo de Soja Vila Velha Pet 900ml - 7896223709423'

]

delta_lag_fim = [
  
'Price BAU 1-4 Cxs - Óleo De Soja Tipo 1 Liza Garrafa 900Ml - 7896036090244', 
'Price BAU 1-4 Cxs - Óleo De Soja Vila Velha Pet 900Ml - 7896223709423',
'Price BAU 1-4 Cxs - Óleo De Soja Soya Garrafa 900Ml - 7891107101621',

 

#'Delta Wind 28D Price BAU 1-4 Cxs - Óleo De Soja Tipo 1 Liza Garrafa 900Ml - 7896036090244' ,
#'Delta Wind 28D Price BAU 1-4 Cxs - Óleo De Soja Vila Velha Pet 900Ml - 7896223709423',
#'Delta Wind 28D Price BAU 1-4 Cxs - Óleo De Soja Soya Garrafa 900Ml - 7891107101621'
 
'Delta Lag- Mean 7/14/21/28 Price BAU 1-4 Cxs - Óleo De Soja Tipo 1 Liza Garrafa 900Ml - 7896036090244' ,
'Delta Lag- Mean 7/14/21/28 Price BAU 1-4 Cxs - Óleo De Soja Vila Velha Pet 900Ml - 7896223709423',
'Delta Lag- Mean 7/14/21/28 Price BAU 1-4 Cxs - Óleo De Soja Soya Garrafa 900Ml - 7891107101621',


'Delta BAU 1-4 Cxs Rio_Atacadao Óleo de Soja Soya Garrafa 900ml - 7891107101621',
'Delta BAU 1-4 Cxs Torre_Cia Óleo de Soja Soya Garrafa 900ml - 7891107101621',

'Delta BAU 1-4 Cxs Guanabara Óleo de Soja Soya Garrafa 900ml - 7891107101621',
#'Delta BAU 1-4 Cxs Nova_Coqueiro_Alimentos Óleo de Soja Soya Garrafa 900ml - 7891107101621',
'Delta BAU 1-4 Cxs Mundial Óleo de Soja Soya Garrafa 900ml - 7891107101621',


'Delta BAU 1-4 Cxs Portal_Eden Óleo de Soja Vila Velha Pet 900ml - 7896223709423',
'Delta BAU 1-4 Cxs Nova_Coqueiro_Alimentos Óleo de Soja Vila Velha Pet 900ml - 7896223709423',
 

'Delta BAU 1-4 Cxs Mundial Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244', 
'Delta BAU 1-4 Cxs Torre_Cia Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244',
'Delta BAU 1-4 Cxs Nova_Coqueiro_Alimentos Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244',
'Delta BAU 1-4 Cxs Rio_Atacadao Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244',
'Delta BAU 1-4 Cxs Portal_Eden Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244',
'Delta BAU 1-4 Cxs Guanabara Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244',
#'Delta BAU 1-4 Cxs Recreio_Distribuidora Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244'


] 
 

lista_correl = lista_bau +   ofertao_columns + delta_lag_fim

lista_correl = lista_bau + delta_lag_mean # +['Price Óleo de Soja Soya Garrafa 900ml - 7891107101621'] + ['Price Óleo de Soja Vila Velha Pet 900ml - 7896223709423'] + ['Price Óleo de Soja Tipo 1 Liza Garrafa 900ml - 7896036090244'] 
lista_correl = lista_bau + delta_conc  + delta_lag_mean +  ofertao_columns + cols_gmv_categoria
 
#lista_correl = lista_bau + delta_lag_mean #
#]lista_correl =  lista_bau + cols_price_clubbi
#df_correl[df_correl.isnull().any(axis=1)]
 
 
#df_correl = df_correl[df_correl.index >  pd.Timestamp('2023-10-01')]   
#df_correl = df_correl[df_correl.index >  pd.Timestamp('2024-01-01')]  
#df_correl[delta_conc]
  

# %% 

#df_correl = df_correl[df_correl['Ofertão ' +]==0]
 
#prod_lista # Price 
#mean_lista #Lag- 
#wind_lista #Wind 
#delta_wind #Delta Wind 
#delta_lag_mean #Delta  
#delta_lag #Delta Lag
#delta_conc #Delta Conc  

 

y = df_correl['Gmv ' + categoria].reset_index(drop = True)
X = df_correl[lista_correl].reset_index(drop = True)

scaler = preprocessing.StandardScaler()
X_scaled = X
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
X_scaled = sm.add_constant(X_scaled)
 
modelo = sm.OLS(y, X_scaled)
modelo_v1 = modelo.fit()
print(modelo_v1.summary())
 
df_regressao = pd.DataFrame(modelo_v1.params,columns=['Coef']).merge(  pd.DataFrame(modelo_v1.pvalues,columns=['P-Valor']), how='left', left_index=True, right_index=True)   
df_regressao['P-Valor'] = df_regressao['P-Valor'].apply(lambda x: round(x, 3))
df_regressao = df_regressao.sort_values('P-Valor',ascending = True)
df_regressao

# conta delta = preço clubbi/preço concorrencia-1 => 
# Clubbi = 4 / Conc = 5 => 4/5-1 = -20% => Delta Negativo = Maior GMV Categoria  
# Clubbi = 6 / Conc = 4 => 6/4 -1 = 50% => Delta Positivo = Menor GMV Categoria 
  
# %% Outro Modelo


y = df_correl['Gmv ' + categoria].reset_index(drop = True)
X = df_correl[lista_correl].reset_index(drop = True)

scaler = preprocessing.StandardScaler()
X_scaled = X
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
X_scaled = sm.add_constant(X_scaled)

size = int(len(y)*0.8)
y_train =  y.iloc[:size]
y_test =  y.iloc[size:]
y_test

x_train = X_scaled.iloc[:size]
x_test = X_scaled.iloc[size:]
x_test
 

modelo = sm.OLS(y_train, x_train)

#modelo = sm.OLS(y, X_scaled)
modelo_v1 = modelo.fit()
print(modelo_v1.summary())
 
df_regressao = pd.DataFrame(modelo_v1.params,columns=['Coef']).merge(  pd.DataFrame(modelo_v1.pvalues,columns=['P-Valor']), how='left', left_index=True, right_index=True)   
df_regressao['P-Valor'] = df_regressao['P-Valor'].apply(lambda x: round(x, 3))
df_regressao = df_regressao.sort_values('P-Valor',ascending = True)
df_regressao
 
valores_previstos = modelo_v1.predict(x_test)

valores_previstos = pd.DataFrame(valores_previstos, columns = ['Predito'])

valores_previstos['Predito'] = np.where((valores_previstos['Predito'] < 0) , 0 ,valores_previstos['Predito'] )
valores_previstos
 
 
df_test = pd.DataFrame(y_test).merge( valores_previstos, how='left', left_index=True, right_index=True)   
df_test

rmse = sqrt(mean_squared_error(df_test['Gmv Categoria_Óleos, Azeites E Vinagres'], df_test['Predito']))
mape =  mean_absolute_percentage_error(df_test['Gmv Categoria_Óleos, Azeites E Vinagres'], df_test['Predito'])

print('rmse')
print(rmse)

print('mape')
print(mape)

 
 

# print(modelo_v1.params)
# valores_previstos = modelo_v1.predict(X)
# valores_previstos


# df_teste = df_regressao.reset_index(drop = False)
# df_teste['1-4'] =  df_teste['index'].str[:var] == 'Delta BAU 1-4 Cxs' 
# df_teste['index'][df_teste['1-4']== True].unique()

 
 
# # %% Scatter Plot 
# sns.regplot(x='%Conversao_Categoria', y="Gmv Categoria", data=df_correl_oleo)
 
# # Range de valores para x e y
# x_range = [df_correl_oleo['%Share_Gmv_Categoria'].min(), df_correl_oleo['%Share_Gmv_Categoria'].max()]
# y_range = [df_correl_oleo['Gmv Categoria'].min(), df_correl_oleo['Gmv Categoria'].max()]



# # Primeira camada do Scatter Plot
# scatter_plot = df_correl_oleo.plot(kind = 'scatter', x = '%Share_Gmv_Categoria', y = 'Gmv Categoria', xlim = x_range, ylim = y_range)

 
# # Segunda camada do Scatter Plot (médias)
# meanY = scatter_plot.plot(x_range, [df_correl_oleo['Gmv Categoria'].mean(),df_correl_oleo['Gmv Categoria'].mean()], '--', color = 'red', linewidth = 1)
# meanX = scatter_plot.plot([df_correl_oleo['%Share_Gmv_Categoria'].mean(),df_correl_oleo['%Share_Gmv_Categoria'].mean()], y_range, '--', color = 'red', linewidth = 1)

# # Terceira camada do Scatter Plot (linha de regressão)
# #regression_line = scatter_plot.plot(df_correl_oleo['RM'], valores_previstos, '-', color = 'orange', linewidth = 2)
 
 


# # Fazendo previsões com o modelo treinado
# RM = 5
# Xp = np.array([1, RM])
# print ("Se RM = %01.f nosso modelo prevê que a mediana da taxa de ocupação é %0.1f" % (RM, modelo_v1.predict(Xp)))


 



# %%

