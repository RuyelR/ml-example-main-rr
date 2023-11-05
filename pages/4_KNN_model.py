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
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code
from Hello import iris_clean_data


def KNN_all_acc_chart(X,Y):
    number_of_iter = 20
    ks=[]
    accs=[]
    best_mean=0
    for num in range(1,number_of_iter+1):
        knn = KNeighborsClassifier(n_neighbors=num)
        cross_val = cross_val_score(knn,X, Y, cv=4, scoring='accuracy')
        ks.append(num)
        accuracy = np.mean(cross_val)*100
        accs.append(accuracy)
        if accuracy>best_mean:
            best_mean = accuracy
            best_k = num
    st.write("Best k: ", best_k, "with best mean: ", best_mean)
    # st.write(ks, accs)
    chart_data = pd.DataFrame({'K neighbors': ks, 'Avg. Accuracy': accs})
    st.line_chart(data=chart_data, x='K neighbors',y='Avg. Accuracy',use_container_width=True)

def KNN_demo():
    weight = st.radio(label='Weight', options=('uniform','distance'), help='Pick which weight you want to test', disabled=False, label_visibility="visible")
    n_val = st.sidebar.select_slider(label="KNN neighbors", options=(1,2,3,4,5,6),value=3,help='Select the number of neighbors for the KNN')
    data = iris_clean_data()
    st.write(data.head())
    X = data.drop(columns='species')
    Y = data['species']
    # Since KNN doesnt accept string Labels we encode it
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.5, random_state=2)
    classifier = KNeighborsClassifier(n_neighbors=n_val, weights=weight)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write('### Confusion matrix:')
    cm
    accuracy = accuracy_score(y_test, y_pred)*100
    st.write(f'# Accuracy of the model with n = {n_val} is equal ' + str(round(accuracy, 2)) + ' %.')
    KNN_all_acc_chart(X,Y)


st.set_page_config(page_title="KNN Model demo", page_icon="📈")
st.markdown("# KNN Model demo")
st.sidebar.header("KNN Model demo")
st.write(
    """This demo illustrates the KNN model utilized over the Iris dataset."""
)

KNN_demo()

show_code(KNN_demo)
