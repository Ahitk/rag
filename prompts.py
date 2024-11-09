from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

## ===================================== GENERATION ========================================== 
# Main prompt for the Telekom IT support chatbot
telekom_template = """
You are a friendly and helpful IT support chatbot designed for question-answering tasks related to telekom.de support, assisting Telekom IT support representatives.

Ensure that your responses are contextually appropriate for the ongoing chat. 
Use the provided context and conversation history to answer questions, always responding in the language in which the question was asked.

If you do not know the answer or if the context lacks necessary information, 
politely inform the user that you cannot assist with their query and redirect them to visit www.telekom.de/hilfe for further support.

If the question is technical, provide sufficient detail to assist Telekom IT Support staff.

Question: {question}
Context: {context}
Conversation history: {chat_history}
Answer:
"""

# Create a prompt template from the above-defined string
# This will be used to dynamically format the chatbot's prompt
prompt_telekom = ChatPromptTemplate.from_template(telekom_template)



## CRAG and Self-RAG: Transform query
# This section defines a prompt structure for rewriting user questions to improve context and clarity.

# SYSTEM: Defines the role of the re-writer and specifies the goals for rephrasing questions.
re_write_chat_system = """
You are a question re-writer that converts an input question into a more effective version optimized for context search and web search. 
The input question is a query directed to a telecom customer support chatbot. 
Analyze the input question to understand the underlying semantic intent or meaning. 
When rephrasing the question, consider the context to ensure clarity and comprehensiveness, including references to the user's previous questions. 
The rewritten question should accurately reflect the user's intent. 
Use specific terms instead of pronouns found in the input question, and include all relevant terms from both the input question and the question history. 
Do not merge the input question with the question history; instead, strengthen the input question alone. 
Always rewrite the question in the same language as the original.
"""

# SYSTEM (Alternate): A shorter variation of the re-writer instructions for other contexts.
re_write_system = """
You are a question re-writer that converts an input question into a better version optimized for context search and web search. 
Always re-write the question in the same language as the original question. Look at the input and try to reason about the underlying semantic intent or meaning.
"""

# ChatPromptTemplate for the re-write functionality
# This structure creates a reusable template for dynamic interaction with the re-writer.
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        # Defining the system's behavior (predefined set of instructions).
        ("system", re_write_chat_system),
        # Example input for the re-writer, with placeholders for dynamic values.
        (
            "human", 
            "The input question given to the chat: \n\n {question} \n\n The question history: {question_history} \n\n"
        ),
    ]
)

## CRAG and Self-RAG: Retrieval grader
# This section defines the logic for evaluating whether a retrieved document is relevant
# to a user's query. The evaluator works on a binary decision: 'yes' (relevant) or 'no' (not relevant).

# SYSTEM: Defines the evaluator's role and decision-making process.
system = """
You are an evaluator assessing whether a retrieved document contains information useful for answering a user's question. 
Your task is to determine if the document includes relevant information that could potentially answer the question. 
The goal is to filter out completely irrelevant documents. 
Respond with a binary answer: either 'yes' or 'no'. 
If the document contains keywords or information that might relate to the question, respond with 'yes'. 
If the document is irrelevant to the question, respond with 'no'.
"""

# ChatPromptTemplate for the retrieval grader
# This structure creates a reusable template for interacting with the evaluator.
grade_prompt = ChatPromptTemplate.from_messages(
    [
        # Defining the system's behavior (predefined set of instructions).
        ("system", system),
        # Example input for the evaluator, using placeholders for dynamic values.
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

## CRAG and Self-RAG: Hallucination Grader Prompt
# System message for hallucination grading
# The system message provides instructions to the LLM (Language Model) for assessing whether its generated response 
# is grounded in or supported by a provided set of facts. The response is evaluated on a binary scale: 
# 'yes' (grounded in the facts) or 'no' (not grounded in the facts).
system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# Define a prompt template for hallucination assessment
# This template structures the interaction with the LLM by specifying how the input facts and the generated 
# output should be presented for grading.
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        # System instruction for the grader
        ("system", system_hallucination),
        # Human input: Provides the set of facts and the generation to be evaluated
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

# ROUTING: Question router prompt 
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.                                    
Use the vectorstore for questions on these topics. For all else, use web-search."""

## ===================================== RETRIEVAL ========================================== 

## Multi-Query: Template for Generating Alternative Questions

multi_query_template = """You are an AI language model chatbot. Your task is to generate four 
different versions of the given user question, and if applicable, to take into account the 
Question history to retrieve relevant documents from a vector database. By generating 
multiple perspectives on the user question, your goal is to help the user overcome some 
of the limitations of distance-based similarity search. 

Provide these alternative questions separated by newlines. Always respond in the language 
in which the question was asked. 

Original question: {question} 
Question history: {question_history}
"""
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
            """You are an expert at understanding and reformulating questions. 
            Your task is to step back and paraphrase a question to a more general and easier-to-answer step-back question. 
            Take into account the chat history and prior questions to ensure coherence. 
            Here are a few examples of this transformation process:""",
        ),
        few_shot_prompt,
        ("user", "{question}"),
        ("system", "Here is the question history to consider: {question_history}"),
    ]
)

stepback_response_prompt_template = """
You are a friendly and helpful chatbot assistant designed to answer questions of Telekom customers related to telekom.de support. 
Your task is to generate a comprehensive yet concise response based on the provided context and prior conversation history. 
Ensure your response is consistent and coherent with the information available. Use both the normal and step-back contexts when applicable to provide the most accurate answer. 
Always respond in the language in which the question was asked.

If you do not know the answer or the provided documents do not contain the necessary information, 
politely state that you cannot assist with this query and redirect the user to www.telekom.de/hilfe for further support.

Keep your answers concise (up to four sentences), but if the response is technical, provide sufficient detail.

# Normal Context: {normal_context}
# Step-Back Context: {step_back_context}
# Original Question: {question}
# Conversation History: {chat_history}
# Answer:
"""

stepback_response_prompt = ChatPromptTemplate.from_template(stepback_response_prompt_template)


## HyDE: Document Generation

# Define a template for generating content.
# The template specifies that the content should be brief, clear, and informative.
hyde_content_template = """You are creating professional and customer-focused web page content and texts for a major telecommunications provider like Telekom.de. 
Your content is very brief, very clear, and informative. Please write a text for the following question and question history:
Question: {question}
Question history: {question_history}
text:"""

# Create a prompt template using the defined template.
# This template will be used to generate content for a given question.
prompt_hyde = ChatPromptTemplate.from_template(hyde_content_template)