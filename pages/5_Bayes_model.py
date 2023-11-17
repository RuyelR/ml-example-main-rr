# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code
from utils import iris_clean_data,accuracy_display,cm_display,user_demo


def Bayes_demo():
    data = iris_clean_data()
    X = data.drop(columns='species')
    Y = data['species']
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.5, random_state=2)
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    
    user_demo(gnb,)
    accuracy_display(y_test, y_pred)
    cm_display(y_test, y_pred)
    
    cross_val = cross_val_score(gnb,X, Y, cv=4, scoring='accuracy')
    st.write("### The  Naive Bayes Classifier average accuracy of crossvalidation cv=4 is equal to "+ str(round(np.mean(cross_val), 2)) + ' %.')
    


st.set_page_config(page_title="Bayes Model demo", page_icon="📈")
st.markdown("# Bayes Model demo")
st.sidebar.header("Bayes Model demo")
st.write(
    """This demo illustrates the Bayes model utilized over the Iris dataset."""
)

Bayes_demo()

show_code(Bayes_demo)
