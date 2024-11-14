import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import pandas as pd
from datetime import datetime
from mistralai_azure import MistralAzure
import re
import streamlit as st


st.set_page_config(
    page_title="Bug Buster Labs - AI Bot",
    page_icon="🔒",
    layout="wide"
)


logging.basicConfig(
    filename='app.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Get current date in the format needed for the prompt
current_date = datetime.now().strftime('%Y-%m-%d')

AZURE_AI_ENDPOINT = st.secrets["AZURE_AI_ENDPOINT"]
AZURE_AI_API_KEY = st.secrets["AZURE_AI_API_KEY"]

# Replace database configuration
DB_CONFIG = {
    "host": st.secrets["HOST"],
    "database": st.secrets["DATABASE_NAME"],
    "user": st.secrets["USER"],
    "password": st.secrets["PASSWORD"],
    "port": st.secrets["PORT"]
}

client = MistralAzure(azure_endpoint=AZURE_AI_ENDPOINT, azure_api_key=AZURE_AI_API_KEY)


def initialize_language_model(query):
    system_message = f"""
    You are an expert data analyst who will help generate precise DataFrame queries from natural language questions. You have access to a security vulnerabilities DataFrame called 'combined_df' with the following columns:
    # Submission Identifiers
    - id_submission          # Unique identifier for submission
    - submissions_id         # Submission reference ID
    - program_id_submission  # Program identifier
    - program_id_program     # Related program identifier

    # Submission Content
    - submission_title       # Title of the vulnerability submission
    - detail_description     # Detailed description of the vulnerability
    - step_to_reproduce      # Steps to reproduce the vulnerability
    - remediation_recommendation  # Recommended fixes
    - target_title          # Title of the target system/component
    - target_url1-5         # Target URLs (up to 5)

    # Severity and Priority
    - severity_submission   # Severity level (Critical, Severe, Moderate, etc.)
    - priority             # Priority level (P1, P2, P3, etc.)
    - vtx                  # Version/tracking number
    - cvss_score           # CVSS vulnerability score

    # Technical Details
    - type_of_testing_allowed      # Type of testing permitted
    - languages_frameworks         # Programming languages/frameworks involved
    - asset_environments_submission # Environment information

    # Status and Tracking
    - submission_status    # Current status of submission
    - assignee_date       # Assignment date
    - tags                # Associated tags

    # Program Information
    - program_type        # Type of program
    - program_package     # Package information
    - program_title       # Title of the program
    - project_description # Description of the project
    - private_program     # Boolean indicating if private
    - project_tags        # Project-related tags

    # Scope Information
    - scope_title         # Title of scope
    - scope_items_url1-5  # Scope URLs (up to 5)
    - out_Of_scope_item_url1-5 # Out of scope URLs (up to 5)

    # Reward Information
    - severity_program    # Program severity level
    - p1_reward_amt      # P1 priority reward amount (float)
    - p2_reward_amt      # P2 priority reward amount (float)
    - p3_reward_amt      # P3 priority reward amount (float)
    - p4_reward_amt      # P4 priority reward amount (float)
    - p5_reward_amt      # P5 priority reward amount
    - maximun_budget     # Maximum budget (float)

    # Dates and Environment
    - start_date         # Start date (datetime64[ns, UTC])
    - end_date           # End date (datetime64[ns, UTC])
    - testing_allowed    # Testing permissions
    - language_framworks # Language frameworks
    - asset_environments_program # Program environment details

    Key Columns:
    submission_title: Title of the vulnerability submission
    detail_description: Detailed description of the vulnerability
    severity_submission: Severity level (Critical, Severe, Moderate)
    priority: Priority level (P1, P2, P3, P4)
    start_date: UTC datetime format (e.g., 2024-09-17 18:30:00+00:00)
    end_date: UTC datetime format (e.g., 2024-12-30 18:30:00+00:00)
    submission_status: Status of the submission
    p1_reward_amt to p4_reward_amt: Reward amounts in float
    target_title: Title of the target system
    program_title: Title of the program
    languages_frameworks: Programming languages used
    asset_environments_submission: Environment information
    cvss_score: Vulnerability score
    tags: Associated tags
    created_at: Shows the dat when submission was created

    Here is the sample row for your reference how it looks like:
        +--------------------------------------+----------------------------+-------------------------------+--------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+------------------+-------+--------+------------+-----------+-------------------------------------+---------------------+---------------------+---------------------+---------------------+---------------------+-------------------------+---------------------------+-------------------------------------+------------------+--------------------+--------------------+--------------------------+------------------------+----------------------------------+------------------------+------------------------+--------------------------------+-------------------------------------+---------------------------+----------------------------------+----------------------------+---------------------------------------+----------------------------+----------------------------------+---------------------+---------------------+----------------------------+-----------------------+----------------------------+---------------------------+--------------------------+-----------------------------+-----------------------------+-----------------------+------------------------------------------+---------------------------+-------------------+----------------------------+--------------------------------------------+----------------------------------------+--------------------+----------------------+
    | id_submission                       | submissions_id             | submission_title              | detail_description                                                                          | step_to_reproduce                                                                             | remediation_recommendation   | severity_submission | priority | vtx        | cvss_score | target_title                     | target_url1         | target_url2         | target_url3         | target_url4         | target_url5         | type_of_testing_allowed | languages_frameworks | asset_environments_submission | submission_status  | assignee_date      | tags               | program_id_submission | program_id_program | program_type       | program_package     | program_title                 | project_description                     | private_program | project_tags       | scope_title        | scope_items_url1   | scope_items_url2   | scope_items_url3   | scope_items_url4   | scope_items_url5   | out_Of_scope_item_url1 | out_Of_scope_item_url2 | out_Of_scope_item_url3 | out_Of_scope_item_url4 | out_Of_scope_item_url5 | severity_program | expected_vulnerability_types | p1_reward_amt  | p2_reward_amt  | p3_reward_amt  | p4_reward_amt  | p5_reward_amt  | maximun_budget  | start_date                   | end_date                     | testing_allowed                                | language_framworks               | asset_environments_program |
    +--------------------------------------+----------------------------+-------------------------------+--------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+------------------+-------+--------+------------+-----------+-------------------------------------+---------------------+---------------------+---------------------+---------------------+---------------------+-------------------------+---------------------------+-------------------------------------+------------------+--------------------+--------------------+--------------------------+------------------------+----------------------------------+------------------------+------------------------+--------------------------------+-------------------------------------+---------------------------+----------------------------------+----------------------------+---------------------------------------+----------------------------+----------------------------------+---------------------+---------------------+----------------------------+-----------------------+----------------------------+---------------------------+--------------------------+-----------------------------+-----------------------------+-----------------------+------------------------------------------+---------------------------+-------------------+----------------------------+--------------------------------------------+----------------------------------------+--------------------+----------------------|
    | 470917a2-0ab4-41fe-9b7c-0dd3db481ffc | BSB000000000028            | Able to view passwords of other researchers | <p>Able to view passwords of other researchers, offline cracking can be done on these hashed passwords. pbkdf2_sha256</p> | <p>View url - https://bugbusters-api.azurewebsites.net/api/v1/user/list_by_role/</p>               | <p>Disable to option to view profiles by all users.</p><p>If needed for admin make sure to not display passwords.</p> | Critical           | P1    | 3.1.15  |            |           | Able to view passwords of all researchers. | https://bugbusters-api.azurewebsites.net/api/v1/user/list_by_role/ |                       |                     |                     | web                     | django                    | prod                 | review               |                     | Known Issue      | 530dfaa9-858b-44bf-972e-15868f2a0661 | BPRG000000000012      | Bug Bounty Program - Public | Expert       | Bugbusterslabs Platform Security testing | <p>Inviting Hackers to test and submit vulnerabilities for our AI-sentinelVDP (Bug bounty Platform)</p> | false             | web application      | Web testing         | https://www.ai-sentinelvdp.com/customer | https://www.ai-sentinelvdp.com/researcher |                       |                     |                     | https://www.bugbusterslabs.com |                       |                       | Critical             | OWASP             | 50.0          | 25.0         | 15.0         | 10.0         |               | 1000.0           | 2024-09-17 18:30:00+00:00 | 2024-12-30 18:30:00+00:00 | OWASP,WEB Application security testing,BlackBox testing | Python,web app | AZURE                  |
    +--------------------------------------+----------------------------+-------------------------------+--------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+------------------+-------+--------+------------+-----------+-------------------------------------+---------------------+---------------------+---------------------+---------------------+---------------------+-------------------------+---------------------------+-------------------------------------+------------------+--------------------+--------------------+--------------------------+------------------------+----------------------------------+------------------------+------------------------+--------------------------------+-------------------------------------+---------------------------+----------------------------------+----------------------------+---------------------------------------+----------------------------+----------------------------------+---------------------+---------------------+----------------------------+-----------------------+----------------------------+---------------------------+--------------------------+-----------------------------+-----------------------------+-----------------------+------------------------------------------+---------------------------+-------------------+----------------------------+----------


    Important Notes:
    1. All text fields are case-insensitive in searches
    2. Dates are in UTC timezone and datetime64[ns, UTC] format - ALWAYS use tz='UTC' in Timestamp creation
    3. Current date to use for relative date queries: {current_date}
    4. Always return minimal relevant columns by default:
   - For general queries: ['submission_title', 'severity_submission', 'priority']
   - For date queries: ['submission_title', 'start_date', 'end_date']
   - For severity queries: ['submission_title', 'severity_submission', 'priority']
   - For reward queries: ['submission_title', 'priority', 'p1_reward_amt']
    ...And so on based on the query

    INSTRUCTIONS:
    - Provide ONLY the DataFrame query code, nothing else
    - Use lowercase for string matching
    - Always include .str.contains(na=False) for text searches
    - For date calculations, ALWAYS include time and timezone: pd.Timestamp('2024-11-14 00:00:00', tz='UTC')
    - Analyse the user question carefully and do not include all the columns include the only columns which are necessary to answer the user question 
    - Avoid generating the dataframe query that adds extra unnecesary columns
    - While generating the query enclose your query in <DF_QUERY> dataframe query </DF_QUERY> tags
    - If the query is unclear and is irrelevant and is totally irrelevant to the context like how's the weather, What is the capital of India etx then return <DF_QUERY> UNKNOWN </DF_QUERY>

    Example User Queries and Expected Response Format:
    Q: "Show critical severity issues from last 3 months"
    A: <DF_QUERY>combined_df[(combined_df['severity_submission'].str.lower() == 'critical') & (combined_df['start_date'] >= pd.Timestamp('2024-11-14', tz='UTC') - pd.DateOffset(months=3))]</DF_QUERY>

    Q: "Find submissions with password in title"
    A: <DF_QUERY>combined_df[combined_df['submission_title'].str.lower().str.contains('password', na=False)]</DF_QUERY>

    Q: "Show P1 vulnerabilities with rewards over 30"
    A: <DF_QUERY>combined_df[(combined_df['priority'] == 'P1') & (combined_df['p1_reward_amt'] > 30)]</DF_QUERY>

    Q: "Show critical severity issues from last 3 months"
    A: <DF_QUERY>combined_df[(combined_df['severity_submission'].str.lower() == 'critical') & (combined_df['start_date'] >= pd.Timestamp('2024-11-14 00:00:00', tz='UTC') - pd.DateOffset(months=3))]</DF_QUERY>

    Q: "Show submissions from last month"
    A: combined_df[(combined_df['created_at'] >= pd.Timestamp('2024-10-14 00:00:00', tz='UTC')) & (combined_df['created_at'] <= pd.Timestamp('2024-11-14 23:59:59', tz='UTC'))][['submission_title', 'created_at']]
    
    Remember to:
    - Return ONLY the DataFrame query
    - Always use case-insensitive string matching
    - Always add tz='UTC' when creating Timestamps
    - Use exact column names as provided above
    - Include na=False in str.contains() operations
    """
    user_message = f"User question: {query}"

    resp = client.chat.complete(messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ], model="azureai")
    
    llm_response = resp.choices[0].message.content
    match = re.search(r'<DF_QUERY>\s*(.*?)\s*</DF_QUERY>', llm_response, re.DOTALL)
    if match:
        extracted_string = match.group(1)
        return extracted_string
    else:
        return None

def summary_llm(query, data):
    system_message = f"""
    You are Bug Buster Labs' Expert Security Analysis Assistant. Your role is to provide clear, actionable insights from vulnerability submission data in a professional and engaging manner.

    Current Date: {current_date}

    When analyzing the data, focus on:
    - Key vulnerability findings and their implications
    - Severity and priority patterns
    - Relevant dates and timelines
    - Security impact and recommendations 

    Data from the database:
    {data}

    Please provide analysis in a natural conversational tone while maintaining professionalism. Your response should flow like a well-structured security report - starting with a brief greeting, followed by key findings, supporting details, and ending with a clear summary.

    If you cannot provide a complete answer, acknowledge what information is missing and explain why.

    Remember to:
    - Be precise and security-focused 
    - Keep explanations clear and actionable
    - Maintain a confident yet approachable tone
    - Highlight critical information naturally in the conversation
    """
    user_message = f"User question: {query}"

    resp = client.chat.complete(messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ], model="azureai")
    
    llm_response = resp.choices[0].message.content
    return llm_response


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)
    

def execute_query(query: str, params: tuple = None) -> Optional[list[Dict[str, Any]]]:
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                result = cur.fetchall()  # Fetch all rows
                logging.info(f"Executed query: {cur.query}")
                return [dict(row) for row in result] if result else []
    except psycopg2.OperationalError as e:
        logging.error(f"Operational error: {e}")
    except psycopg2.ProgrammingError as e:
        logging.error(f"Programming error: {e}")
    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
    return []


def fetch_customer_programs(customer_id):
    query = """
    SELECT 
        id, 
        program_id, 
        program_type, 
        program_package, 
        program_title, 
        project_description, 
        private_program, 
        project_tags, 
        scope_title, 
        scope_items_url1, 
        scope_items_url2,
        scope_items_url3, 
        scope_items_url4, 
        scope_items_url5, 
        "out_Of_scope_item_url1", 
        "out_Of_scope_item_url2", 
        "out_Of_scope_item_url3", 
        "out_Of_scope_item_url4", 
        "out_Of_scope_item_url5",
        severity, 
        expected_vulnerability_types, 
        p1_reward_amt, 
        p2_reward_amt, 
        p3_reward_amt, 
        p4_reward_amt, 
        p5_reward_amt, 
        maximun_budget, 
        start_date, 
        end_date, 
        testing_allowed,
        language_framworks, 
        asset_environments
    FROM 
        program_programs
    WHERE 
        customer_id = %s;
    """
    program_details = execute_query(query, (customer_id,))
    
    return pd.DataFrame(program_details)

def fetch_program_ids(customer_id):
    query = """
    select id from program_programs where customer_id = %s;
    """
    program_ids = execute_query(query, (customer_id,))
    return pd.DataFrame(program_ids)

def fetch_program_submissions(customer_id):
    query = """
    select 
        id, 
        submissions_id, 
        submission_title, 
        detail_description, 
        step_to_reproduce, 
        remediation_recommendation,
        severity, 
        priority, 
        vtx, 
        cvss_score, 
        target_title, 
        target_url1, 
        target_url2, 
        target_url3, 
        target_url4, 
        target_url5,
        type_of_testing_allowed, 
        languages_frameworks, 
        asset_environments, 
        created_at,
        submission_status, 
        assignee_date, 
        tags 
    FROM 
        submission_submission 
    WHERE 
        program_id_id=%s;
    """
    submission_details = execute_query(query, (customer_id,))
    return pd.DataFrame(submission_details)

def fetch_all_submissions_for_customer(customer_id):
    program_ids_df = fetch_program_ids(customer_id)
    
    all_submissions = []
    
    for program_id in program_ids_df['id']:
        submission_df = fetch_program_submissions(program_id)
        if not submission_df.empty:
            submission_df['program_id'] = program_id  
            all_submissions.append(submission_df)
    
    if all_submissions:
        combined_df = pd.concat(all_submissions, ignore_index=True)
        return combined_df
    else:
        logging.info(f"No submissions found for customer {customer_id}")
        return pd.DataFrame()  


def combine_programs_and_submissions(customer_id):
    # Fetch program data
    programs_data = fetch_customer_programs(customer_id)
    
    # Fetch all submissions data for the customer
    all_submissions_df = fetch_all_submissions_for_customer(customer_id)
    
    # If no submissions data found, return an empty DataFrame
    if all_submissions_df.empty:
        logging.info(f"No submission data found for customer {customer_id}")
        return pd.DataFrame()
    
    # Merge the two DataFrames on 'program_id'
    combined_df = pd.merge(
        all_submissions_df, 
        programs_data, 
        how='left', 
        left_on='program_id', 
        right_on='id', 
        suffixes=('_submission', '_program')
    )
    
    combined_df.drop(columns=['id_program'], inplace=True)
    
    return combined_df


def save_combined_data_to_csv(combined_df):
    file_name = f"combined_data.csv"
    
    combined_df.to_csv(file_name, index=False)
    logging.info(f"Combined data saved to {file_name}")


def main():
    customer_id = "383efe69-4787-425c-a3ed-2815cf2a3ee3"
    
    # Combine program data and submission data for the customer
    combined_df = combine_programs_and_submissions(customer_id)
    query = "List all the submissions made during last 6 month"
    res = initialize_language_model(query)
    print(res)
    if res != "UNKNOWN":
        dataframe_query_response = eval(res)
        print(dataframe_query_response)
        summary = summary_llm(query, dataframe_query_response)
        print(summary)
    else:
        print("Sorry Please Specify your query properly")
           
    if not combined_df.empty:
        save_combined_data_to_csv(combined_df)
    else:
        print("No data available for the given customer.")



# Main title
st.title("🔒 Bug Buster Labs - Customer Dashboard")

# Configuration in main screen
st.subheader("Configuration")
customer_id = st.text_input("Customer ID", value="383efe69-4787-425c-a3ed-2815cf2a3ee3")

# Function to combine programs and submissions data
def load_data(customer_id):
    try:
        # Your existing data loading logic here
        combined_df = combine_programs_and_submissions(customer_id)
        return combined_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Natural language query section
st.header("📝 Natural Language Query")
query = st.text_input("Enter your query (e.g., 'Show critical severity issues from last 3 months')")

if query:
    with st.spinner("Processing query..."):
        # Get DataFrame query from LLM
        df_query = initialize_language_model(query)
        
        if df_query != "UNKNOWN":
            try:
                # Load the data
                combined_df = load_data(customer_id)
                if combined_df is not None:
                    # Execute the query
                    result_df = eval(df_query)
                    
                    # Display results
                    st.subheader("Query Results")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Export button
                    if not result_df.empty:
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="vulnerability_analysis.csv",
                            mime="text/csv"
                        )
                    
                    # Analysis Summary below results
                    st.subheader("Analysis Summary")
                    if not result_df.empty:
                        summary = summary_llm(query, result_df)
                        st.write(summary)
                    else:
                        st.info("No data found matching your query.")
            
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("Please specify your query properly. Try to be more specific about what vulnerability information you're looking for.")

