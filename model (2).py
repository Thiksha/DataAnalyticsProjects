# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:05:48 2021
@author: sushanth
"""
# from controller import *
import collections
import json
import pandas as pd
from lifetimes.utils import *
from flask import Flask, request
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
# from flask_json_syslog.flask_json_syslog import json
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# user_list = []
def test_api():
    uploaded_file = request.files['document']
    # uploaded_file = 'C:/Users/sushanth/OneDrive/Desktop/treosoft/orders8.csv'
    df = pd.read_csv(uploaded_file)
    df['Name'] = df['Name'].str.strip()
    df.dropna(axis=0, subset=['catId'], inplace=True)
    df['catId'] = df['catId'].astype('str')
    df = df[~df['catId'].str.contains('C')]
    basket = (df
              .groupby(['orderId', 'Name'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('orderId'))

    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket_sets = basket.applymap(encode_units)
    frequent_itemsets = apriori(basket_sets, min_support=0.000498, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    output = rules.sort_values(by=['lift'], ascending=False)
    output = rules.drop(['leverage', 'conviction', 'antecedent support', 'consequent support'], axis=1, inplace=True)
    output = rules.sort_values(by=['confidence'], ascending=False)
    result = rules.drop_duplicates(subset=['lift'], keep='first')
    result = rules.drop_duplicates(subset=['support'], keep='first')
    print(result.sort_values(by=['support'], ascending=False))
    result.sort_values(by=['support'], ascending=False)
    user_list = []
    k = collections.OrderedDict()
    k['Message'] = str('item Data')
    k['Result'] = str('Success')
    combos_list = []
    datatuple = list(result.sort_values(by=['support'], ascending=False).itertuples(index=False))
    for x in datatuple:
        for z in list(x[0]):
            l1 = z
        for z in list(x[1]):
            l2 = z
        l3 = x[2]
        l4 = x[3]
        l5 = x[4]
        t = collections.OrderedDict()
        t['item1'] = l1
        t['item2'] = l2
        t['support'] = l3
        t['confidence'] = l4
        t['lift'] = l5
        combos_list.append(t)
    k["items_list"] = combos_list
    user_list.append(k)
    final = json.dumps(user_list[0], indent=3, sort_keys=False)
    return final


def test_api1():
    uploaded_file = request.files['document1']
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d %H:%M").dt.date

    lf = summary_data_from_transaction_data(df, 'Product Name', 'Date', monetary_value_col='Amount',
                                            observation_period_end='2021-09-21')
    lf.reset_index().head(100)
    bgf = BetaGeoFitter(penalizer_coef=1.0)
    bgf.fit(lf['frequency'], lf['recency'], lf['T'])
    t = 100
    lf['pred_num_txn'] = round(
        bgf.conditional_expected_number_of_purchases_up_to_time(t, lf['frequency'], lf['recency'], lf['T']), 2)
    print(lf['pred_num_txn'])
    # lf.sort_values(by='pred_num_txn', ascending=False).head(10).reset_index()

    shortlisted_customers = lf[lf['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0)
    ggf.fit(shortlisted_customers['frequency'], shortlisted_customers['monetary_value'])
    Data = round(ggf.customer_lifetime_value(
        bgf,
        lf['frequency'],
        lf['recency'],
        lf['T'],
        lf['monetary_value'],
        time=24,
        discount_rate=0.00
    ), 2)
    Data_variable = str(Data).split()
    user_list = []
    k = collections.OrderedDict()
    k['Message'] = str('item Data')
    k['Result'] = str('Success')
    k["items_list"] = Data_variable
    user_list.append(k)
    # print(json.dumps(user_list[0], indent=3, sort_keys=False))
    final1 = json.dumps(user_list[0], indent=3, sort_keys=False)
    return final1

