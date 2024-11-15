import streamlit as st
import os
from datetime import datetime
import pandas as pd
from mistralai_azure import MistralAzure
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import json 
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Bug Buster Labs Customer Bot",
    page_icon="🐞",
    layout="wide"
)

logging.basicConfig(
    filename='app.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize session state
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = "383efe69-4787-425c-a3ed-2815cf2a3ee3"  # Default customer ID

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


def base_llm(query):
    system_message = """
    You are an intelligent AI query classifier tasked with categorizing user queries into three main categories:

    DATABASE_AGENT: If the user's query involves retrieving information that requires database access (e.g., "Show me critical vulnerabilities," "Show submission status," or "Show reports"), classify it as DATABASE_AGENT.

    GENERAL_AGENT: If the user's query pertains to cybersecurity or technical concepts (e.g., "How can I optimize my bug bounty scope for better results?" or "What is an XSS attack?"), classify it as GENERAL_AGENT.

    NONE: If the user's query is unrelated to cybersecurity, database access, or technical topics (e.g., "What is the capital of France?" or "What is a cheeseburger?"), classify it as NONE.

    Output Format: Strictly return your classification in the following format:
    <BASE>Your response</BASE>
    eg : 
    Question: Show me vulnerability reports
    Output: <BASE> DATABASE_AGENT </BASE> 
    Question: What is MITM Attack
    Output: <BASE> GENERAL_AGENT </BASE>
    Question: List top 10 movies
    Output: <BASE> NONE </BASE> 
    """
    user_message = f"User question: {query}"

    resp = client.chat.complete(messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ], model="azureai")
    
    llm_response = resp.choices[0].message.content
    match = re.search(r'<BASE>\s*(.*?)\s*</BASE>', llm_response, re.DOTALL)
    if match:
        extracted_string = match.group(1)
        return extracted_string
    else:
        return None


def general_agent(query):
    system_message = """
    You are Bug Buster Labs Smart AI Assistant, an intelligent AI designed to provide precise and accurate answers to any cybersecurity-related questions. Your goal is to respond to user queries concisely and informatively, using your expertise in cybersecurity concepts, best practices, tools, and techniques.

    Focus on providing clear, direct answers while avoiding unnecessary details. If the question is ambiguous, ask for clarification.
    """
    user_message = f"User question: {query}"

    resp = client.chat.complete(messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ], model="azureai")
    
    llm_response = resp.choices[0].message.content
    return llm_response


def initialize_language_model(query):

    resolve_query = r"""<DF_QUERY> combined_df.groupby(['submission_status', 'severity_submission']).agg({'submission_title': 'count'}).reset_index().rename(columns={'submission_title': 'count'})</DF_QUERY>"""
    average_time_query = r"""<DF_QUERY>combined_df.assign(created_at=pd.to_datetime(combined_df['created_at']).dt.tz_localize(None)).query("submission_status == 'resolved'").assign(resolution_time=(pd.Timestamp.now() - pd.to_datetime(combined_df['created_at']).dt.tz_localize(None))).agg({'resolution_time': 'mean'})</DF_QUERY>"""
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

    Here are the datatypes of Columns it will help you generating the dataframe query:
        id_submission                       object 
    1   submissions_id                      object 
    2   submission_title                    object 
    3   detail_description                  object 
    4   step_to_reproduce                   object 
    5   remediation_recommendation          object 
    6   severity_submission                 object 
    7   priority                            object 
    8   vtx                                 object 
    9   cvss_score                          float64
    10  target_title                        object 
    11  target_url1                         object 
    12  target_url2                         float64
    13  target_url3                         float64
    14  target_url4                         float64
    15  target_url5                         float64
    16  type_of_testing_allowed             object 
    17  languages_frameworks                object 
    18  asset_environments_submission       object 
    19  created_at                          object 
    20  submission_status                   object 
    21  assignee_date                       float64
    22  tags                                object 
    23  program_id_submission               object 
    24  program_id_program                  object 
    25  program_type                        object 
    26  program_package                     object 
    27  program_title                       object 
    28  project_description                 object 
    29  private_program                     bool   
    30  project_tags                        object 
    31  scope_title                         object 
    32  scope_items_url1                    object 
    33  scope_items_url2                    object 
    34  scope_items_url3                    float64
    35  scope_items_url4                    float64
    36  scope_items_url5                    float64
    37  out_Of_scope_item_url1              object 
    38  out_Of_scope_item_url2              float64
    39  out_Of_scope_item_url3              float64
    40  out_Of_scope_item_url4              float64
    41  out_Of_scope_item_url5              float64
    42  severity_program                    object 
    43  expected_vulnerability_types        object 
    44  p1_reward_amt                       float64
    45  p2_reward_amt                       float64
    46  p3_reward_amt                       float64
    47  p4_reward_amt                       float64
    48  p5_reward_amt                       float64
    49  maximun_budget                      float64
    50  start_date                          object 
    51  end_date                            object 
    52  testing_allowed                     object 
    53  language_framworks                  object 
    54  asset_environments_program          object 


    Key Columns:
    submission_title: Title of the vulnerability submission
    detail_description: Detailed description of the vulnerability
    severity_submission: Severity level (Critical, Severe, Moderate)
    priority: Priority level (P1, P2, P3, P4)
    start_date: UTC datetime format (e.g., 2024-09-17 18:30:00+00:00)
    end_date: UTC datetime format (e.g., 2024-12-30 18:30:00+00:00)
    submission_status: Status of the submission either approved/rejected    
    p1_reward_amt to p4_reward_amt: Reward amounts in float
    target_title: Title of the target system
    program_title: Title of the program
    languages_frameworks: Programming languages used
    asset_environments_submission: Environment information
    cvss_score: Vulnerability score
    tags: Associated tags
    created_at: Shows when submission was created

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
    5.Sentra AI is something that decided whther the submissions are rejected or approved the status of submissions approved/rejected/pendingare present in submission_status column so if user asks anything related to Sentra AI or submission status make sure to include this column from dataframe
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


    Example User Queries and Expected Response Format:
    Q: "Show critical severity issues from last 3 months"
    A: <DF_QUERY>combined_df[(combined_df['severity_submission'].str.lower() == 'critical') & (combined_df['start_date'] >= pd.Timestamp('2024-11-14', tz='UTC') - pd.DateOffset(months=3))]</DF_QUERY>

    Q: "Find submissions with password in title"
    A: <DF_QUERY>combined_df[combined_df['submission_title'].str.lower().str.contains('password', na=False)]</DF_QUERY>

    Q: "Show P1 vulnerabilities with rewards over 30"
    A: <DF_QUERY>combined_df[(combined_df['priority'] == 'P1') & (combined_df['p1_reward_amt'] > 30)]</DF_QUERY>

    Q: "Show critical severity issues from last 3 months"
    A: <DF_QUERY>combined_df[(combined_df['severity_submission'].str.lower() == 'critical') & (combined_df['start_date'] >= pd.Timestamp('2024-11-14 00:00:00', tz='UTC') - pd.DateOffset(months=3))]</DF_QUERY>

    Q: "Show submissions from last month (Make sure you put use the current date provided for this don't putany random date check the current date and add the current date accordingly)" 
    A: <DF_QUERY>combined_df[(combined_df['created_at'] >= pd.Timestamp('2024-10-14 00:00:00', tz='UTC')) & (combined_df['created_at'] <= pd.Timestamp('2024-11-14 23:59:59', tz='UTC'))][['submission_title', 'created_at']]</DF_QUERY>
    
    Q: "For summary of resolved and unresolved vulnerabilities:"
    A: {resolve_query}

    Q: "For status Summary"
    A: <DF_QUERY>combined_df.groupby(['submission_status', 'severity_submission'])['submission_title'].count().reset_index()</DF_QUERY>

    Q: "Critical vulnerabilities based on provided timespan from user again use the current date provided to put the proper timestamps
    A: <DF_QUERY>combined_df[(pd.to_datetime(combined_df['created_at'], utc=True) >= pd.Timestamp('2024-11-07 00:00:00', tz='UTC')) & (pd.to_datetime(combined_df['created_at'], utc=True) <= pd.Timestamp('2024-11-14 23:59:59', tz='UTC')) & (combined_df['severity_submission'].str.lower().isin(['critical', 'severe']))][['submission_title', 'severity_submission', 'priority', 'created_at']]</DF_QUERY>

    Q: For "Which areas of my system are most vulnerable?"
    A: <DF_QUERY>combined_df.groupby(['target_title', 'severity_submission'])['submission_title'].count().reset_index().sort_values('submission_title', ascending=False)</DF_QUERY>

    Q: How many vulnerabilities have been reported in the last 30 days? (Make sure to refer the current date provided)
    A: <DF_QUERY> len(combined_df[combined_df["created_at"].astype(str).str[:10] >= "2024-10-16"]) </DF_QUERY>

    Q: What is the average time to resolve reported vulnerabilities?
    A: {average_time_query}

    Q: What percentage of submissions submitted were resolved? (Make sure to refer this query if question revolves around percantage of solved/resolved/pending etc..)
    A: <DF_QUERY> (combined_df['submission_status'].eq('resolved').mean() * 100).round(2) </DF_QUERY>

    #Below Queries are more focuses towards generating Reports refer them for creating differnet types of Reports based on query incorporating the current date provided

    Q: Generate a monthly report summarizing all vulnerabilities
    A: <DF_QUERY> combined_df.assign(month=pd.to_datetime(combined_df['created_at']).dt.strftime('%Y-%m')).groupby(['month', 'severity_submission', 'submission_status'])['submission_title'].count().reset_index().sort_values(['month', 'severity_submission'], ascending=[True, True]) </DF_QUERY>

    Q: Show me the trends for vulnerabilities over the last quarter
    A: <DF_QUERY> combined_df.assign(date=pd.to_datetime(combined_df['created_at'])).sort_values('date').assign(quarter=pd.to_datetime(combined_df['created_at']).dt.to_period('Q')).groupby(['quarter', 'severity_submission'])['submission_title'].count().reset_index().sort_values(['quarter', 'severity_submission'], ascending=[True, True]) </DF_QUERY>

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
    You are Bug Buster Labs' Expert Security Analysis Assistant. Your role is to provide clear answer to user question.
    Your task is to simply synthesis the response to user question 
    You are provided with iuser question and data extracted from smart Database system that corrosponds to user 's question 
    Everytime you get the data just answer it don't ask to provide more data and i can't answer as data is insufficient
    The data you recevie is sufficient to answer the user question just properly summarise it to user

    Current Date: {current_date}

    Data from the Smart Database:
    {data}

    Please provide analysis in a natural conversational tone while maintaining professionalism. Your response should flow like a well-structured security report - starting with a brief greeting, followed by key findings, supporting details, and ending with a clear summary.

    Remember to:
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

DB_CONFIG = {
    "host": os.environ.get('HOST'),
    "database": os.environ.get('DATABASE_NAME'),
    "user": os.environ.get('USER', 'bugbuster_admin'),
    "password": os.environ.get('PASSWORD'),
    "port": os.environ.get('PORT'),
}


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


def generate_response(customer_id, query):
    base_agent_response = base_llm(query)

    if base_agent_response == "DATABASE_AGENT":

        # Combine program data and submission data for the customer
        combined_df = combine_programs_and_submissions(customer_id)

        res = initialize_language_model(query)
        print(res)
            
        try:
            dataframe_query_response = eval(res)
        except Exception as e:
            logging.error(f"Error evaluating query: {e}")
            dataframe_query_response = None
            
        if dataframe_query_response is not None:
            summary = summary_llm(query, dataframe_query_response)
            return summary
        else:
            error_msg = "Error processing your query. Please try again."
            return error_msg
    
    elif base_agent_response == "GENERAL_AGENT":
        general_agent_response = general_agent(query)
        return general_agent_response
    
    elif base_agent_response == "NONE":
        error_msg = "I am sorry i cannot help you with this"
        return error_msg
    
    else:
        error_msg = "Something went wrong please try again later"
        return error_msg


st.title("🐞 Bug Buster Labs Dashboard")

# Sidebar
with st.sidebar:
    st.header("Settings")
    customer_id = st.text_input("Customer ID", value=st.session_state.customer_id)
    st.session_state.customer_id = customer_id
    
    st.markdown("---")
    st.markdown("### Query Examples")
    st.markdown("""
    - Show critical severity issues
    - Show submissions from last month
    - Generate a monthly report
    - What is the average time to resolve vulnerabilities?
    - Show P1 vulnerabilities with rewards over 30
    """)

# Main content
st.header("Bug Bounty Query Interface")

# Query input
query = st.text_area("Enter your query:", height=100)

if st.button("Submit Query", type="primary"):
    with st.spinner("Processing your query..."):
        try:
            response = generate_response(st.session_state.customer_id, query)
            
            # Display response in a nice format
            st.markdown("### Results")
            
            # Check if response is a DataFrame
            if isinstance(response, pd.DataFrame):
                st.dataframe(response)
                
                # Add download button for DataFrame
                csv = response.to_csv(index=False)
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            else:
                st.markdown(response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error processing query: {str(e)}")
