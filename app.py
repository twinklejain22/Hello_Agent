import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables from .env (fallback if user doesn't type a key in sidebar)
load_dotenv()

# ---- System prompt for the AI agent ----
SYSTEM_PROMPT = """You are a strict data assistant. You answer questions ONLY by inspecting and analyzing the pandas DataFrame(s) provided to you. You must follow these rules at all times:

1. ALWAYS INSPECT THE DATA FIRST
Before answering any question, inspect the relevant DataFrame(s) using pandas — check columns, data types, and actual values. Never guess what the data might contain. Base every answer strictly on what you find by inspecting the DataFrame(s).

2. NEVER USE GENERAL KNOWLEDGE
Do not answer from prior/general knowledge, assumptions, or anything outside the provided data. If the answer cannot be found in the DataFrame(s), say so clearly — do not make up or infer an answer from outside sources.

3. PREFER DESCRIPTIVE TEXT COLUMNS
When a question calls for a text-based or descriptive answer, prioritize columns named (or similar to) "Answer", "Policy", "Description", or "Details" if they exist in the data. Use these columns first before considering others.

4. USE PROPER OPERATIONS FOR NUMERIC QUESTIONS
For numeric questions (totals, averages, counts, min/max, etc.), use the correct pandas operation — sum(), mean(), count(), min(), max(), etc. — on the appropriate column(s). Never estimate or approximate a numeric answer; compute it directly from the data.

5. ANSWER IN SIMPLE, CLEAR ENGLISH WITH ACTUAL VALUES
Give your final answer in plain, simple English. Always quote the actual value(s) found in the data (e.g., exact numbers, exact text) rather than vague or generalized statements. Keep answers concise and directly tied to the data you inspected.

If a question cannot be answered from the available data, respond with: "I cannot find this information in the provided data."
"""

# ---- Page setup ----
st.set_page_config(page_title="Chat With Your CSV", layout="wide")

# ---- Sidebar ----
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")

# Use sidebar key if provided, otherwise fall back to .env
api_key = api_key_input if api_key_input else os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.sidebar.warning("Enter your OpenAI API key to get started.")

# ---- Main area ----
st.title("Chat With Your CSV")
st.write("Upload one or more CSV files, then ask questions about your data in plain English.")

# File uploader (accepts multiple files)
uploaded_files = st.file_uploader(
    "Upload CSV file(s)",
    type=["csv"],
    accept_multiple_files=True
)

# Store dataframes in a list, and keep filenames in a matching list
dataframes = []
filenames = []

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        dataframes.append(df)
        filenames.append(file.name)

    st.subheader("Preview")
    for name, df in zip(filenames, dataframes):
        st.markdown(f"**{name}**")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

        # Quick sanity checks so you can confirm the data looks correct
        with st.expander(f"Column info for {name}"):
            st.write("Column names:", list(df.columns))
            st.write("Data types:")
            st.write(df.dtypes)
            if df.isnull().sum().sum() > 0:
                st.warning(f"This file has {df.isnull().sum().sum()} missing values.")

# ---- Question input ----
st.subheader("Ask a question about your data")
user_question = st.text_area("Your question", placeholder="e.g. What is the average value in column X?")

# ---- Answer section ----
st.subheader("Answer")
answer_placeholder = st.empty()

if st.button("Get Answer"):
    if not api_key:
        st.error("Please provide your OpenAI API key first.")
    elif not dataframes:
        st.error("Please upload at least one CSV file first.")
    elif not user_question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            # ---- Step 6: Create the data frame agent ----
            llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4o-mini",
                temperature=0  # low temperature so answers are stable
            )

            agent = create_pandas_dataframe_agent(
                llm,
                dataframes,               # list of dataframes passed to the agent
                verbose=True,
                allow_dangerous_code=True, # permission to run pandas operations
                max_iterations=30,         # give it more steps before giving up
                max_execution_time=60,     # seconds, generous timeout
                handle_parsing_errors=True # don't crash on a malformed intermediate step
            )

            # ---- Step 7: Combine system prompt + user question into one final query ----
            final_query = f"{SYSTEM_PROMPT}\n\nQuestion: {user_question}"

            try:
                response = agent.invoke(final_query)
                answer_placeholder.success(response["output"])
            except Exception as e:
                answer_placeholder.error(f"Something went wrong: {e}")