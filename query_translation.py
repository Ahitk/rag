
### RAG-Fusion

# Create a chain for generating four related search queries
generate_fusion_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

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

