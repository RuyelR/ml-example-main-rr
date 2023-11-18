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

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import streamlit as st
from streamlit.hello.utils import show_code
from utils import iris_clean_data,user_demo,accuracy_display,cm_display

def KNN_demo():
    weight = st.radio(label='Weight', options=('uniform','distance'), help='Pick which weight you want to test', disabled=False, label_visibility="visible")
    n_val = st.sidebar.select_slider(label="KNN neighbors", options=(1,2,3,4,5,6),value=3,help='Select the number of neighbors for the KNN')
    data = iris_clean_data()
    X = data.drop(columns='species')
    Y = data['species']
    # Since KNN doesnt accept string Labels we encode it
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.5, random_state=2)
    classifier = KNeighborsClassifier(n_neighbors=n_val, weights=weight)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    
    
    user_demo(classifier, labelen=le)
    accuracy_display(y_test, y_pred)
    cm_display(y_test, y_pred)


st.set_page_config(page_title="KNN Model demo", page_icon="📈")
st.markdown("# KNN Model demo")
st.sidebar.header("KNN Model demo")
st.write(
    """This demo illustrates the KNN model utilized over the Iris dataset."""
)

KNN_demo()


show_code(KNN_demo)
