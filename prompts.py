from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

## Main prompt: telekom assistant
telekom_template = """
You are a friendly and helpful chatbot assistant designed for question-answering tasks related to telekom.de support, providing assistance to both Telekom IT Support employees and experts, as well as Telekom customers.

Use the provided context and the conversation history to answer the questions. Always respond in the language in which the question was asked.

If you don't know the answer or if the provided documents do not contain the necessary information, simply state that you cannot assist with this query and kindly redirect the user to visit www.telekom.de/hilfe for further support.

Keep your answers concise (up to four sentences), but if the response is technical, provide sufficient detail to assist Telekom IT Support staff.

Question: {question}
Context: {context}
Conversation history: {chat_history}
Answer:
"""
prompt_telekom = ChatPromptTemplate.from_template(telekom_template)

#yedek main prompt:
'''
## Main prompt: telekom assistant
telekom_template = """You are an assistant for question-answering tasks for telekom.de help, providing answers to Telekom customers or potential customers. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer or if the provided documents do not contain relevant information, simply say that unfortunately, you cannot assist with this question and please visit www.telekom.de/hilfe for further assistance. 
Use up to four sentences and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt_telekom = ChatPromptTemplate.from_template(telekom_template)'''

## CRAG and Self-RAG: retrieval grader
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

## CRAG and Self-RAG: re_write prompt
re_write_system = """You are a question re-writer that converts an input question into a better version optimized for web search and context search.\n 
     Always provide the question in German. Look at the input  and try to reason about the underlying semantic intent or meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", re_write_system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

## CRAG and Self-RAG: hallucination grader prompt
system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_hallucination),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

## CRAG and Self-RAG: answer grader prompt
grader_system = """You are a grader assessing whether an answer corresponds to a question or whether it is an appropriate response to that question.\n
    Give a binary score of ‘yes’ or ‘no’. ‘yes’ means that the answer corresponds to the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

## Multi-Query: Template for Generating Alternative Questions
multi_query_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Always respond in the language in which the question was asked.
Original question: {question}"""
# Create a prompt template for generating multiple queries of the user's question
multi_query_prompt = ChatPromptTemplate.from_template(multi_query_template)


## RAG-Fusion: template for generating multiple search queries based on a single input query.
fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
# Create a prompt template for generating multiple queries of the user's question
prompt_rag_fusion = ChatPromptTemplate.from_template(fusion_template)

## Step Back
# Few Shot Examples
few_shot_examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]

# Transform examples into example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=few_shot_examples,
)

step_back_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ]
)

## HyDE: Document Generation
# This section is responsible for creating professional and customer-focused content
# for a major telecommunications provider based on a given question.

# Define a template for generating content.
# The template specifies that the content should be brief, clear, and informative.
hyde_content_template = """You are creating professional and customer-focused web page content and texts for a major telecommunications provider like Telekom.de. 
Your content is very brief, very clear, and informative. Please write a text for the following question:
Question: {question}
text:"""

# Create a prompt template using the defined template.
# This template will be used to generate content for a given question.
prompt_hyde = ChatPromptTemplate.from_template(hyde_content_template)

## Decomposition: Sub-questions prompt
decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(decomposition_template)

# Decomposition answer recursion
decomposition_template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""
decomposition_prompt = ChatPromptTemplate.from_template(decomposition_template)

# Decomposition individual answer prompt
decomposition_individual_template = """Here is a set of Q+A pairs:

{decomposition_individual_context}

Use these to synthesize an answer to the question: {question}
"""

decomposition_individual_prompt = ChatPromptTemplate.from_template(decomposition_individual_template)
