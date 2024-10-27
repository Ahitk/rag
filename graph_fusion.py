import operator
import streamlit as st
from IPython.display import Image, display
from tavily import TavilyClient
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document
from langgraph.graph import END, StateGraph, START
from langchain_community.callbacks import get_openai_callback
from typing_extensions import TypedDict
from typing import List, Annotated
from pprint import pprint
from indexing import get_vectorstore
import routing
import initials
import prompts

### Tavily web search tool
tavily_client = TavilyClient(api_key = initials.TAVILY_API_KEY)

### Retrieval Grader
# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

### Hallucination Grader
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

### Answer Grader
# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call for retrieval
structured_llm_grader = initials.model.with_structured_output(GradeDocuments)
# LLM with function call for hallucination
structured_llm_hallucination_grader = initials.model.with_structured_output(GradeHallucinations)
# LLM with function call for answer
structured_llm_answer_grader = initials.model.with_structured_output(GradeAnswer)

structured_llm_router = initials.model.with_structured_output(routing.RouteUserQuery)

retrieval_grader = prompts.grade_prompt | structured_llm_grader
hallucination_grader = prompts.hallucination_prompt | structured_llm_hallucination_grader
answer_grader = prompts.answer_prompt | structured_llm_answer_grader
question_rewriter = prompts.re_write_prompt | initials.model | StrOutputParser()


# SUAN ICIN TAM OLARAK NEREDE KULLANACAGIMI BILMIYORUM
# Re-write question ve append new web search context
#grader_docs = retriever.get_relevant_documents(question)
#doc_txt = " ".join([doc.page_content for doc in grader_docs])


# === GRAPH ===

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question : str # User question
    generation : str # LLM generation
    web_search : str # Binary decision to run web search
    max_retries : int # Max number of retries for answer generation 
    answers : int # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents : List[str] # List of retrieved documents
    chat_history : list
    question_history : list

### Nodes


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    chat_history = state["chat_history"]
    question_history = state["question_history"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question, "question_history": question_history })
    print("\tTransformed question: ", better_question)
    return {"question": better_question, 
            "chat_history": chat_history, 
            "question_history": question_history,
            "documents": documents}

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    question_history = state["question_history"]

    vector_store = get_vectorstore(question, initials.model, initials.data_directory, initials.embedding)
    retriever = vector_store.as_retriever()

    # Generate multiple queries using the multi_query_prompt and model
    generate_multi_queries = (
        prompts.multi_query_prompt
        | initials.model 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    # Retrieval
    # Generate the multiple queries based on user input
    #multiple_queries = generate_multi_queries.invoke({"question": question, "question_history": question_history})
    retrieval_chain_rag_fusion = generate_multi_queries | retriever.map() | initials.reciprocal_rank_fusion
    fusion_docs = retrieval_chain_rag_fusion.invoke({"question": question, "question_history": question_history})
    #formatted_docs = initials.format_fusion_docs_with_similarity(fusion_docs, question)
    
    return {"documents": fusion_docs, "question": question, "question_history": question_history}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]  # fusion_docs burada geçiyor
    
    filtered_docs = []
    web_search = "No"
    
    for idx, doc_tuple in enumerate(documents, start=1):  # enumerate ile dokümanları numaralandırıyoruz
        document, similarity_score = doc_tuple  # tuple'ı Document ve score olarak ayırıyoruz
        # Document içeriğini değerlendiriyoruz
        score = retrieval_grader.invoke({"question": question, "document": document.page_content})
        grade = score.binary_score.strip().lower()  # strip() gereksiz boşlukları kaldırır
    
        
        # "yes", "1", ya da "true" gibi yanıtları uygun olarak değerlendiriyoruz
        if grade in ["yes", "1", "true", 1]:  
            print(f"GRADE: Document-{idx} RELEVANT")
            filtered_docs.append(document)  # yalnızca Document nesnesini ekliyoruz
        else:
            print(f"GRADE: Document-{idx} NOT RELEVANT")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    loop_step = state.get("loop_step", 0)

    fusion_rag_chain = (prompts.prompt_telekom | initials.model | StrOutputParser())

    with get_openai_callback() as cb:
        generation = fusion_rag_chain.invoke({
            "context": documents, 
            "question": question,
            "chat_history": chat_history
        }) if documents else "No relevant documents found."


    # Return the updated state with generation
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "loop_step": loop_step + 1,
    }

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
  
    question = state["question"]
    documents = state["documents"]

    # Web search
    web_search_tool = tavily_client.qna_search(question)
    web_results = Document(page_content=web_search_tool)
    documents.append(web_results)
    #print(documents)
    return {"documents": documents, "question": question}

### Edges

def route_question(state):
    """
    Route question to web search or RAG 

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    source = structured_llm_router.invoke([SystemMessage(content=routing.query_router_instructions)] + [HumanMessage(content=state["question"])]) 
    if source.datasource == 'websearch':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    print(web_search)
    state["documents"]
    #state["filtered_docs"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: SOME DOCUMENTS ARE NOT RELEVANT TO QUESTION, WEB SEARCH---")
        return "web_search_node"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3) # Default to 3 if not provided


    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("DECISION: GENERATION ADDRESSES QUESTION")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("DECISION: GENERATION DOES NOT ADDRESS QUESTION")
            return "not useful"
        else:
            print("DECISION: MAX RETRIES REACHED")
            return "max retries" 
    elif state["loop_step"] <= max_retries:
        print("DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY")
        return "not supported"
    else:
        print("DECISION: MAX RETRIES REACHED")
        return "max retries" 
 

def run_graph(question, chat_history, question_history, documents):
    
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search_node", web_search)  # web search

    # Build graph
    # workflow.set_entry_point("transform_query") bu sekilde de grafik'e baslanabilir, alttakiyle ayni.
    workflow.add_edge(START, "transform_query")
    workflow.add_conditional_edges(
        "transform_query",    
        route_question,
        {
            "websearch": "web_search_node",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search_node": "web_search_node",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search_node", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "web_search_node",
            "max retries": END,
        },
    )

    # Compile
    app = workflow.compile()

    # Run
    inputs = {"question": question, "chat_history": chat_history, "question_history": question_history, "documents": documents}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"CURRENT GRAPH NODE: '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("===========================================================")

    #display(Image(app.get_graph().draw_mermaid_png()))
    print("---END!---")
    # Final generation
    answer = value["generation"]
    docs = value["documents"]
    question = value["question"]
    documents = initials.format_docs(docs, question)

    return answer, documents
