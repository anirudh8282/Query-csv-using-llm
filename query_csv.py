import pandas as pd
import streamlit as st
import json
from langchain import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="QUERY CSV")
st.title("Query csv")

st.write("Please enter your OpenAI API key.")
openai_api_key = st.text_input("OpenAI API Key", type="password")

def csv_tool(filename: str, openai_api_key: str):
    df = pd.read_csv(filename)
    print("CSV loaded successfully")
    return create_pandas_dataframe_agent(
        OpenAI(api_key=openai_api_key, temperature=0), 
        df, 
        verbose=True, 
        allow_dangerous_code=True  # This line enables the dangerous code execution
    )

def ask_agent(agent, query):
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        3. If a line chart is more appropriate, your reply should look like this:
           {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Note: We only accommodate two types of charts: "bar" and "line".

        4. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        For example:
           {"answer": "The Product with the highest Orders is '15143Exfo'"}

        5. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

        Now, let's tackle the query step by step. Here's the query for you to work on: 
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
            df = pd.DataFrame({col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])})
            index_column = data['columns'][0]  # Assuming the first column can be used as the index
            df.set_index(index_column, inplace=True)
            st.bar_chart(df)
        except (ValueError, KeyError) as e:
            print(f"Error creating bar chart: {e}")
            st.write(f"Error creating bar chart: {e}")

    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df = pd.DataFrame({col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])})
            index_column = data['columns'][0]  # Assuming the first column can be used as the index
            df.set_index(index_column, inplace=True)
            st.line_chart(df)
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
        # Create an agent from the CSV file.
        agent = csv_tool(data, openai_api_key)
        # Query the agent.
        response = ask_agent(agent=agent, query=query)
        # Decode the response.
        decoded_response = decode_response(response)
        # Write the response to the Streamlit app.
        write_answer(decoded_response)
    else:
        st.write("Please upload a CSV file.")
