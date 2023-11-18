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

import altair as alt
import streamlit as st
from streamlit.logger import get_logger
from utils import iris_clean_data

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Welcome to ML model demo",
        page_icon="👋",
    )

    st.write("# Welcome to my machine learning algorithm demo! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Welcome to my first app demonstrating three different machine learning algorithm approaches to the Iris dataset.
    """
    )
    data = iris_clean_data()

    if st.sidebar.toggle(label="Check Data",value=False) != False:
        n_rows = st.sidebar.slider(label="Number of rows", min_value=1,max_value=5,value=3)
        st.write("### Sample of dataset:")
        st.write("First %s rows: "%n_rows, data.head(n_rows))
        start_index = (len(data) - n_rows) // 2
        st.write("Middle %s rows:"%n_rows, data.iloc[start_index:start_index + n_rows])
        st.write("Last %s rows:"%n_rows, data.tail(n_rows))
    if st.sidebar.toggle(label="Describe data",value=False) != False:
        st.write("### Describing all features: ", data.describe()) 
    Sdtframe = data.groupby('species')
    st.write("Mean: \n",Sdtframe.mean())
    st.write("\nMedian/50% quantile: \n", Sdtframe.quantile().pivot_table(columns="species",))
    st.scatter_chart(data=data,x='sepal_length', y='sepal_width',size=30,use_container_width=True)
    st.bar_chart(data=data, x='petal_length', y='petal_width', use_container_width=True)


if __name__ == "__main__":
    run()
