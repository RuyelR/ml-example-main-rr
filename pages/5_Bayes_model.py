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

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing  import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code
from Hello import iris_clean_data


def Bayes_demo():
    data = iris_clean_data()
    st.write('Dataset: ')   
    st.write(data.head())
    X = data.drop(columns='species')
    Y = data['species']
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.5, random_state=2)
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write('### Confusion matrix:')
    cm
    accuracy = accuracy_score(y_test, y_pred)*100
    st.write(f'# Accuracy of the model is equal to ' + str(round(accuracy, 2)) + ' %.')
    cross_val = cross_val_score(gnb,X, Y, cv=4, scoring='accuracy')
    st.write("### Naive Bayes Classifier Accuracy crossvalidation cv=4", cross_val)
    st.write("### The  Naive Bayes Classifier average accuracy of crossvalidation cv=4 is equal to "+ str(round(np.mean(cross_val), 2)) + ' %.')
    


st.set_page_config(page_title="Bayes Model demo", page_icon="📈")
st.markdown("# Bayes Model demo")
st.sidebar.header("Bayes Model demo")
st.write(
    """This demo illustrates the Bayes model utilized over the Iris dataset."""
)

Bayes_demo()

show_code(Bayes_demo)
