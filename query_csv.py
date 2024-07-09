import pandas as pd
import streamlit as st
import json
from langchain import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="QUERY CSV")
st.title("Query CSV")

st.write("Please enter your OpenAI API key.")
openai_api_key = st.text_input("OpenAI API Key", type="password")

def csv_tool(filename: str, openai_api_key: str):
    df = pd.read_csv(filename)
    print("CSV loaded successfully")
    return create_pandas_dataframe_agent(
        OpenAI(api_key=openai_api_key, temperature=0), 
        df, 
        verbose=True, 
        allow_dangerous_code=True
    )

def ask_agent(agent, query):
    prompt = (
        """
        You are working with a CSV file and need to respond to queries about the data. 

        Here are the response formats:

        1. If the query requires a table, respond with:
           {"table": {"columns": ["column1", "column2", ...], "data": [["value1", "value2", ...], ["value1", "value2", ...], ...]}}

        2. For a bar chart, respond with:
           {"bar": {"columns": ["column1"], "data": [["label1", value1], ["label2", value2], ...]}}

        3. For a line chart, respond with:
           {"line": {"columns": ["column1"], "data": [["label1", value1], ["label2", value2], ...]}}

        4. For a plain question that doesn't need a chart or table, respond with:
           {"answer": "Your answer"}

        5. If the answer is not known, respond with:
           {"answer": "I do not know."}

        Ensure all string values in the "columns" list and data list are encased in double quotes.

        Now, handle the following query step by step: 
        """
        + query
    )

    try:
        response = agent.run(prompt)
        print("Agent response received")
    except Exception as e:
        print(f"Error during agent run: {e}")
        raise

    return str(response)

def decode_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return {"answer": "Error decoding response"}

def write_answer(response_dict: dict):
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.bar_chart(df.set_index(data["columns"][0]))
        except (ValueError, KeyError) as e:
            print(f"Error creating bar chart: {e}")
            st.write(f"Error creating bar chart: {e}")

    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.line_chart(df.set_index(data["columns"][0]))
        except (ValueError, KeyError) as e:
            print(f"Error creating line chart: {e}")
            st.write(f"Error creating line chart: {e}")

    if "table" in response_dict:
        data = response_dict["table"]
        try:
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.table(df)
        except ValueError as e:
            print(f"Error creating table: {e}")
            st.write(f"Error creating table: {e}")

st.write("Please upload your CSV file below.")
data = st.file_uploader("Upload a CSV", type="csv")
query = st.text_area("Send a Message")

if st.button("Submit Query", type="primary"):
    if data is not None:
        # Save the uploaded CSV file to a path
        with open("/mnt/data/uploaded_file.csv", "wb") as f:
            f.write(data.getbuffer())
        # Create an agent from the CSV file.
        agent = csv_tool("/mnt/data/uploaded_file.csv", openai_api_key)
        # Query the agent.
        response = ask_agent(agent=agent, query=query)
        # Decode the response.
        decoded_response = decode_response(response)
        # Write the response to the Streamlit app.
        write_answer(decoded_response)
    else:
        st.write("Please upload a CSV file.")
