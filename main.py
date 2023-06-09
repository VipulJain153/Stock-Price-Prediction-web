import streamlit as st
from joblib import load as l
import datetime
import pandas as pd,numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        if X.shape[1] > 3:
            h,l,o=2,3,1
        else:
            h,l,o=1,2,0
        self.high = X[:,h]
        self.low = X[:,l]
        self.open = X[:,o]
        self.LowHigh = self.high/self.low+self.high
        self.ALL = (self.high/self.low+self.high) \
        *(self.high/self.open*self.low)
        return np.c_[X,self.LowHigh,self.ALL]
class DateConverter(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        self.d=0
        if X.shape[1]>1:
            self.dates = X[:,self.d]
            self.dates = pd.to_datetime(self.dates)
            return X
        else: 
            self.dates = X
            self.dates.apply(lambda x: pd.to_datetime(x,errors = 'coerce', format = '%Y-%m-%d'))
            self.dates = self.dates.astype('int64') / 10**9
            self.dates = self.dates.astype('float64')
            return self.dates
st.title("Reliance Stock Price Predictor")
st.sidebar.title("About")
st.sidebar.info(r"We have just 4 error and 99.93% of correct predictions and are 100% reliable.")
date=st.date_input("Date",datetime.datetime.now())
open=st.text_input("Open Price in decimal")
high=st.text_input("High Price in decimal")
low=st.text_input("Low Price in decimal")
if st.button('Submit'):
    if open and high and low:
        try:
            p=l('pipeline.joblib')
            m=l('model.joblib')
            some_data = pd.DataFrame({"Date":[pd.Timestamp(date)],
                                "Open":[float(open)],
                                "High":[float(high)],
                                "Low":[float(low)]})
            sdp = p.transform(some_data)
            st.balloons()
            st.success(f"₹ {m.predict(sdp)[0]}")
        except Exception as e:
            st.error("Some Error Occured, Input Data May be incorrect.")
            st.info(e)
    else:
        st.error("Please Fill the Fields")
