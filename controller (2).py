# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:11:23 2021

@author: sushanth
"""

import  model 
from flask import *

app = Flask(__name__)

@app.route('/mba', methods=['POST'])
def mba():
    x=model.test_api()
    return x

@app.route("/pltv",methods=['POST'])
def pltvs():
    y=model.test_api1()
    return y


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)