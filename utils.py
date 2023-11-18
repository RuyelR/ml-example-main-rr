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

import inspect
import textwrap
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score

import streamlit as st

def user_guide():
    Sdtframe = iris_clean_data().groupby('species')
    st.write("Median/50% quantile: (use as guide)")
    st.write(Sdtframe.quantile().pivot_table(columns="species",))

def accuracy_display(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)*100
    st.write(f'## Accuracy of the model = ' +":green[%s]"%(str(round(accuracy, 2))) + ' %.')
    pass

def cm_display(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    st.write('### Confusion matrix:')
    st.write(cm)


def iris_clean_data():
    data_file = 'Iris_Data.csv'
    data = pd.read_csv(data_file)
    data["species"] = data["species"].str.removeprefix("Iris-")
    return data

def user_demo(classifier, labelen=None):
    data = iris_clean_data()
    X = data.drop(columns='species')
    # User input
    ptl = st.text_input(label="Petal length",value="1.4")
    ptw = st.text_input(label="Petal width",value="0.2")
    spl = st.text_input(label="Sepal length",value="5")
    spw = st.text_input(label="Sepal width",value="3")
    # Predictions
    flower_stat = pd.DataFrame(data=[[spl, spw, ptl, ptw]],columns= X.columns)
    flower_pred = classifier.predict(flower_stat)
    if labelen:
        flower_pred = labelen.inverse_transform(flower_pred)
    st.markdown("# The predicted result is: " + f":violet[%s]"%flower_pred[0])

def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", value=False)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))
