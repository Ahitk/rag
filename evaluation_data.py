# Import necessary libraries
import os
import time
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_community.callbacks import get_openai_callback
from indexing import get_vectorstore_semantic
import prompts as prompts
import initials as initials
import evaluation
import chromadb

"""
TEST_DIRECTORIES:

"Vertrag & Rechnung",
"Hilfe bei Störungen",
"Mobilfunk",
"Internet & Telefonie",
"TV",
"MagentaEINS",
"Apps & Dienste",
"Geräte & Zubehör"
"""

test_directory = '/Users/taha/Desktop/rag/data/Vertrag & Rechnung'
# Define input CSV path
input_csv_path = '/Users/taha/Desktop/rag/data/Vertrag & Rechnung/_testset_semantic.csv'  # Input CSV file path

# Define output CSV path including the filename
output_csv_path = '/Users/taha/Desktop/rag/data/Vertrag & Rechnung/_evaluation_semantic_fusion.csv'  # Output file will be created here

# Function to create the output CSV file at the beginning
def initialize_output_csv(output_path):
    # Directly create the file with the correct header
    with open(output_path, 'w') as file:
        header = (
            "Question,Response,Contexts,Ground Truth,"
            "Token Count,Total Cost (USD),Completion Tokens,Number of Retrieved documents,"
            "Response time,answer_relevancy,context_precision,"
            "context_recall,faithfulness,BleuScore,RougeScore\n"
        )
        file.write(header)
    print(f"Created output file at: {output_path}")

# Function to get response with error handling
def get_response(user_input):
    try:
        # Load vector store and retriever
        vector_store, category = get_vectorstore_semantic(user_input, test_directory, initials.embedding)
        retriever = vector_store.as_retriever()
        
        # Generate multiple queries using the multi_query_prompt and model
        generate_multi_queries = (
            prompts.multi_query_prompt 
            | initials.model 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        # Generate the multiple queries based on user input
        multiple_queries = generate_multi_queries.invoke({"question": user_input, "question_history": []})

        retrieval_chain_rag_fusion = generate_multi_queries | retriever.map() | initials.reciprocal_rank_fusion

        fusion_docs = retrieval_chain_rag_fusion.invoke({"question": user_input, "question_history": []})
        formatted_docs = initials.format_fusion_docs_with_similarity(fusion_docs, user_input)
        docs = [doc for doc, _ in fusion_docs]

        fusion_rag_chain = (prompts.prompt_telekom | initials.model | StrOutputParser())

        # Use OpenAI callback to track costs and tokens
        with get_openai_callback() as cb:
            response = fusion_rag_chain.invoke({
                "context": fusion_docs, 
                "question": user_input,
                "chat_history": []
            }) if fusion_docs else "No relevant documents found."

        # Update total tokens and cost, completion tokens
        total_tokens = cb.total_tokens
        total_cost = cb.total_cost
        completion_tokens = cb.completion_tokens

        return response, formatted_docs, docs, category, total_cost, total_tokens, completion_tokens

    except FileNotFoundError:
        print("Documents could not be loaded. Please check the data directory path.")
        return None, None, None, None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, None, None

# Function to save evaluation data to CSV
def save_evaluation_to_csv(evaluation_data, filename):
    df = pd.DataFrame([evaluation_data])
    df.to_csv(filename, mode='a', index=False, header=False)

# Main execution
def run_evaluations_from_csv(input_csv, output_csv):
    # Directly create the output CSV file with headers at the beginning
    initialize_output_csv(output_csv)

    # Load questions from the CSV file
    questions_df = pd.read_csv(input_csv)
    
    for index, row in questions_df.iterrows():
        user_query = row['question']
        start_time = time.time()  # Start timing
        print(f"Processing question {index + 1}/{len(questions_df)}: {user_query}")

        try:
            # Get the response, generated queries, and retrieved documents
            response, documents, context, category, total_cost, total_tokens, completion_tokens = get_response(user_query)
            print("==========   ANSWER GENERATED  ==========")

            # Initialize metrics_results
            metrics_results = None

            print("==========   EVALUATION  ==========")
            # Evaluate metrics and retrieve dataset
            metrics_results, dataset = evaluation.evaluate_result(user_query, response, context, category)
            print(f"Metrics for question '{user_query}': {metrics_results}")

            if response:
                # Calculate response time
                response_time = time.time() - start_time
                # Clear the system cache after processing the response
                chromadb.api.client.SharedSystemClient.clear_system_cache()

                # Prepare data for CSV
                if metrics_results is not None:
                    # Extract contexts and ground_truth from the dataset
                    contexts = dataset["contexts"][0]  # Access first row's 'contexts'
                    ground_truth = dataset["ground_truth"][0]  # Access first row's 'ground_truth'
                    
                    evaluation_data = {
                        'Question': user_query,
                        'Response': response,
                        'Contexts': contexts,
                        'Ground Truth': ground_truth,
                        'Token Count': total_tokens,
                        'Total Cost (USD)': total_cost,
                        'Completion Tokens': completion_tokens,
                        'Number of Retrieved documents': len(context),
                        'Response time': response_time,
                        'answer_relevancy': metrics_results.get('answer_relevancy'),
                        'context_precision': metrics_results.get('context_precision'),
                        'context_recall': metrics_results.get('context_recall'),
                        'faithfulness': metrics_results.get('faithfulness'),
                        'BleuScore': metrics_results.get('bleu_score'),
                        'RougeScore': metrics_results.get('rouge_score'),

                    }

                    # Save the evaluation data to CSV
                    save_evaluation_to_csv(evaluation_data, output_csv)
                    print(f"Evaluation metrics saved for question '{user_query}'.")

            print("==========   PROCESS ENDED  ==========\n")

        except ValueError as ve:
            print(f"ValueError for question {index + 1}: {ve}")
            print("Skipping to the next question...\n")

        except Exception as e:
            print(f"Unexpected error for question {index + 1}: {e}")
            print("Skipping to the next question...\n")


# Run evaluations
run_evaluations_from_csv(input_csv_path, output_csv_path)