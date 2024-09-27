
### Multi-query

# Define a pipeline for generating alternative queries
generate_multi_queries = (
    multi_query_prompt 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))  # Split the generated output into individual queries
)

def get_unique_union(documents):
    """
    Returns a unique union of retrieved documents.

    This function takes a list of lists of documents, flattens it, and removes duplicates
    to ensure each document is unique.

    Args:
        documents (list of lists): A list where each element is a list of documents.

    Returns:
        list: A list of unique documents.
    """
    # Flatten the list of lists of documents
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Remove duplicates by converting to a set and then back to a list
    unique_docs = list(set(flattened_docs))
    # Deserialize the documents back into their original form
    return [loads(doc) for doc in unique_docs]


### RAG-Fusion

# Create a chain for generating four related search queries
generate_fusion_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


# Function for Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Applies Reciprocal Rank Fusion (RRF) to combine multiple lists of ranked documents.
    
    Parameters:
    - results (list[list]): A list of lists where each inner list contains ranked documents.
    - k (int): An optional parameter for the RRF formula, default is 60.
    
    Returns:
    - list: A list of tuples where each tuple contains a document and its fused score.
    """
    
    # Initialize a dictionary to store the fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Serialize the document to a string format to use as a key
            doc_str = dumps(doc)
            # Initialize the document's score if not already present
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Update the document's score using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort documents based on their fused scores in descending order
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples
    return reranked_results


### HyDE

# Define a chain to generate documents for retrieval.
# This chain uses the prompt template, a language model, and an output parser.
generate_docs_for_retrieval = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()
)

# Run HyDE document generation to produce content for the given question.
# The try-except block handles potential errors during document generation.
try:
    hyde_output = generate_docs_for_retrieval.invoke({"question": question})
    print(f"HyDE hypothetical context:\n{hyde_output.strip()}\n")
except Exception as e:
    logger.error(f"Error generating documents for retrieval: {e}")
    raise


### Step-back

# Generate step-back queries
generate_queries_step_back = step_back_prompt | model | StrOutputParser()
step_back_question = generate_queries_step_back.invoke({"question": question})

print(f"Original Question: {question}")
print(f"Step-Back Question: {step_back_question}")

# Response prompt template
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# Normal Context:
{normal_context}

# Step-Back Context:
{step_back_context}

# Original Question: {question}

# Answer:
"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

### Decomposition
# Chain
generate_queries_decomposition = ( prompt_decomposition | model | StrOutputParser() | (lambda x: x.split("\n")))

# Run
decomposition_questions = generate_queries_decomposition.invoke({"question":question})

