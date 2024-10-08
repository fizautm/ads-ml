import streamlit as st
import pandas as pd

st.title('🤖 Agile Data Science')

st.write('This is app builds a machine learning model')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
df
