import streamlit as st
from ui.components import sidebar_ui, main_ui
from backend.database import init_db


st.set_page_config(page_title='Research Insight Assistant', layout='wide')


# Initialize DB (creates tables if not exist)
init_db()


sidebar_ui()
main_ui()