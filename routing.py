from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from typing import Literal

# Data model
class RouteQuery(BaseModel):
    """Route a user question to the most relevant datacategory."""

    datacategory: Literal["vertrag_rechnung_ihre_daten_kundencenter_login-daten_rechnung_lieferstatus", 
                          "hilfe_stoerungen_stoerungen_selbst_beheben_melden_status_verfolgen",
                          "mobilfunk_tarife_optionen_mobiles-internet_mailbox_esim_sim-karten",
                          "internet_telefonie:_ausbau,_sicherheit,_einstellungen,_bauherren,_glasfaser_und_wlan",
                          "tv_magentatv_streaming-dienste_magentatv_jugendschutz_pins",
                          "magentains_kombi-pakete_mit_magentains_vorteil_und_treuebonus",
                          "apps_dienste_e-mail_magenta_apps_voicemail_app_mobilityconnect",
                          "geraete_zubehoer_anleitungen_fuer_smartphones_tablets_telefone_router_receiver"] = Field(
        ...,
        description="Given a user question choose which datacategory would be most relevant for answering their question",
    )

# Data model
class RouteUserQuery(BaseModel):
    """ Route a user query to the most relevant datasource. """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Prompt 
query_router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vector store contains extensive customer support and help documents related to telecommunications.

These documents cover categories such as: Devices & Accessories, Help with Disruptions, Internet & Telephony, MagentaEINS, Mobile Communications, TV, Contract & Billing, and Apps & Services, among others.

Use the vector store for questions on these telecommunications topics. For all other queries, use web search."""


def define_router(model):
        # LLM with function call 
    structured_model = model.with_structured_output(RouteQuery)

    # Prompt 
    routing_template = """You are an expert at routing user questions to the appropriate data category.

    Based on the help category the question is referring to, route it to the relevant data category. 
    """

    routing_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", routing_template),
            ("human", "{question}"),
        ]
    )

    # Define router 
    router = routing_prompt | structured_model
    return router

def choose_route(result):
    # Kategorileri ve ilgili alt dizinleri bir sözlükte tanımlayın
    category_map = {
        "vertrag_rechnung_ihre_daten_kundencenter_login-daten_rechnung_lieferstatus": "Vertrag & Rechnung",
        "hilfe_stoerungen_stoerungen_selbst_beheben_melden_status_verfolgen": "Hilfe bei Störungen",
        "mobilfunk_tarife_optionen_mobiles-internet_mailbox_esim_sim-karten": "Mobilfunk",
        "internet_telefonie:_ausbau,_sicherheit,_einstellungen,_bauherren,_glasfaser_und_wlan": "Internet & Telefonie",
        "tv_magentatv_streaming-dienste_magentatv_jugendschutz_pins": "TV",
        "magentains_kombi-pakete_mit_magentains_vorteil_und_treuebonus": "MagentaEINS",
        "apps_dienste_e-mail_magenta_apps_voicemail_app_mobilityconnect": "Apps & Dienste",
        "geraete_zubehoer_anleitungen_fuer_smartphones_tablets_telefone_router_receiver": "Geräte & Zubehör"
    }
    
    # Datacategory'yi küçült ve sözlükte ara, yoksa "Others" döner
    return category_map.get(result.datacategory.lower(), "Others")

def get_specific_directory(question, model, data_directory):
    # orjinal rag_advanced satiri:
    # full_chain = router | RunnableLambda(choose_route)
    full_chain = define_router(model) | RunnableLambda(choose_route)
    sub_directory = full_chain.invoke({"question": question})
    # orjinal rag_advanced satiri sildim:
    #print(sub_directory)
    specific_directory = os.path.join(data_directory, sub_directory)
    # orjinal rag_advanced satiri sildim:
    #print(specific_directory)
    print(f"==========   SELECTED DATA CATEGORY: {sub_directory}   ==========")
    return specific_directory