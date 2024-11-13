import os
from typing import Literal
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field

# Define a data model for routing user questions to the most relevant data category
class RouteQuery(BaseModel):
    """
    Data model for categorizing a user question into one of the predefined data categories.

    Attributes:
        datacategory: A predefined category to which the user question is routed. 
                      This helps in determining the most relevant area of information for the query.
    """

    datacategory: Literal[
        # Data categories covering different topics
        "vertrag_rechnung_ihre_daten_kundencenter_login-daten_rechnung_lieferstatus", 
        "hilfe_stoerungen_stoerungen_selbst_beheben_melden_status_verfolgen",
        "mobilfunk_tarife_optionen_mobiles-internet_mailbox_esim_sim-karten",
        "internet_telefonie:_ausbau,_sicherheit,_einstellungen,_bauherren,_glasfaser_und_wlan",
        "tv_magentatv_streaming-dienste_magentatv_jugendschutz_pins",
        "magentains_kombi-pakete_mit_magentains_vorteil_und_treuebonus",
        "apps_dienste_e-mail_magenta_apps_voicemail_app_mobilityconnect",
        "geraete_zubehoer_anleitungen_fuer_smartphones_tablets_telefone_router_receiver"
    ] = Field(
        ...,
        description=(
            "Given a user question, choose the most relevant data category for answering it. "
            "Each category corresponds to a specific type of user inquiry."
        ),
    )

# Define a data model for routing user questions to a data source
class RouteUserQuery(BaseModel):
    """
    Data model for routing a user query to the most appropriate data source.

    Attributes:
        datasource: The data source to which the user query is routed. 
                    Choices are 'vectorstore' or 'websearch'.
    """

    datasource: Literal[
        "vectorstore",  # Refers to a vector store for semantic or embedding-based searches
        "websearch"     # Refers to a standard web search engine
    ] = Field(
        ...,
        description=(
            "Given a user question, decide whether to route it to a 'vectorstore' for semantic "
            "retrieval or 'websearch' for a general web search."
        ),
    )

# Define the routing instructions for the query router
query_router_instructions = """
You are an expert at routing a user question to a vectorstore or web search.

The vector store contains extensive IT support and help documents related to telecommunications.

These documents cover categories such as: 
    - Devices & Accessories
    - Help with Disruptions
    - Internet & Telephony
    - MagentaEINS
    - Mobile Communications
    - TV
    - Contract & Billing
    - Apps & Services

Use the vector store for questions on these telecommunications topics. 
For all other queries, use web search.
"""

def define_router(model):
    """
    Function to define a query router using a provided language model (LLM).
    
    Args:
        model: The base language model to be used for routing queries.

    Returns:
        router: A pipeline that combines a routing prompt and the language model 
                with structured output to route user questions to appropriate data categories.
    """

    # Step 1: Wrap the model to output structured data based on RouteQuery
    # This ensures that the model produces responses in a predefined structured format.
    structured_model = model.with_structured_output(RouteQuery)

    # Step 2: Define the routing prompt template
    # This prompt provides the language model with the context and instructions for routing queries.
    routing_template = """You are an expert at routing user questions to the appropriate data category.

    Based on the help category the question is referring to, route it to the relevant data category. 
    """

    # Step 3: Create a prompt template using ChatPromptTemplate
    # The template includes system-level instructions and a placeholder for the human's question.
    routing_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", routing_template),  # System message provides the routing instructions
            ("human", "{question}"),      # Placeholder for the user's question
        ]
    )

    # Step 4: Define the router pipeline
    # Combines the routing prompt with the structured model to create the query router.
    router = routing_prompt | structured_model

    # Step 5: Return the router
    return router

def choose_route(result):
    """
    Function to map a given `datacategory` to a predefined category or return "Others" 
    if no match is found.

    Args:
        result (object): An object containing the `datacategory` attribute.
    
    Returns:
        str: A corresponding category name from the category_map dictionary or "Others" if no match is found.
    """
    
    # Define a dictionary to map specific subcategories to broader category names
    category_map = {
        # Key: Subcategory name (lowercase expected for matching)
        # Value: Broader category name
        "vertrag_rechnung_ihre_daten_kundencenter_login-daten_rechnung_lieferstatus": "Vertrag & Rechnung",
        "hilfe_stoerungen_stoerungen_selbst_beheben_melden_status_verfolgen": "Hilfe bei Störungen",
        "mobilfunk_tarife_optionen_mobiles-internet_mailbox_esim_sim-karten": "Mobilfunk",
        "internet_telefonie:_ausbau,_sicherheit,_einstellungen,_bauherren,_glasfaser_und_wlan": "Internet & Telefonie",
        "tv_magentatv_streaming-dienste_magentatv_jugendschutz_pins": "TV",
        "magentains_kombi-pakete_mit_magentains_vorteil_und_treuebonus": "MagentaEINS",
        "apps_dienste_e-mail_magenta_apps_voicemail_app_mobilityconnect": "Apps & Dienste",
        "geraete_zubehoer_anleitungen_fuer_smartphones_tablets_telefone_router_receiver": "Geräte & Zubehör"
    }
    
    # Convert the datacategory attribute to lowercase for case-insensitive matching
    # Retrieve the mapped category from the dictionary; default to "Others" if not found
    return category_map.get(result.datacategory.lower(), "Others")

def get_specific_directory(question, model, data_directory):
    """
    Determines the specific subdirectory within a data directory 
    based on a given question, using a model and routing logic.

    Args:
        question (str): The user's question or query.
        model: The machine learning model or logic used for routing.
        data_directory (str): The base directory containing all possible subdirectories.

    Returns:
        str: The full path to the specific subdirectory selected based on the question.
    """
    
    # Original line in rag_advanced:
    # full_chain = router | RunnableLambda(choose_route)
    
    # Define the routing logic by combining the router and route chooser
    full_chain = define_router(model) | RunnableLambda(choose_route)
    
    # Use the defined routing chain to determine the subdirectory based on the question
    sub_directory = full_chain.invoke({"question": question})
    
    # Original debug print from rag_advanced (commented out):
    # print(sub_directory)
    
    # Construct the full path to the specific subdirectory
    specific_directory = os.path.join(data_directory, sub_directory)
    
    # Original debug print from rag_advanced (commented out):
    # print(specific_directory)
    
    # Log the selected data category for visibility
    print(f"==========   SELECTED DATA CATEGORY: {sub_directory}   ==========")
    
    return specific_directory