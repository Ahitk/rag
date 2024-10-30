import pandas as pd
from datasets import Dataset
from langchain.schema import Document
from ragas import evaluate
from ragas.metrics import (   
    answer_relevancy,    
    context_precision,
    context_recall,
    faithfulness,
    BleuScore,
    RougeScore,
)
from ragas.dataset_schema import SingleTurnSample
import asyncio
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# NLTK için gerekli kaynakları indirin
nltk.download('punkt')
nltk.download('punkt_tab')

def evaluate_result(question, answer, contexts, path):
    # Test verisini belirtilen yoldan yükle
    csv_path = f"{path}/_testset_semantic.csv"
    testdata = pd.read_csv(csv_path)
    
    # Sorunun doğru cevabını bul
    ground_truth_row = testdata[testdata["question"] == question]
    if ground_truth_row.empty:
        raise ValueError("The specified question does not exist in the test data.")
    
    true_ground_truth = ground_truth_row.iloc[0]["ground_truth"] 
    
    # Contextleri metin olarak dönüştür
    contexts = [str(context) if isinstance(context, Document) else context for context in contexts]
    
    # Veriyi Dataset formatına dönüştür
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],  
        "ground_truth": [true_ground_truth],
    }
    
    dataset = Dataset.from_dict(data)

    # Diğer metriklerle değerlendir
    evaluation_results = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy,    
            context_precision,
            context_recall,
            faithfulness,
        ],
    )

    results_df = evaluation_results.to_pandas()
    results = {
        "answer_relevancy": results_df["answer_relevancy"].iloc[0],
        "context_precision": results_df["context_precision"].iloc[0],
        "context_recall": results_df["context_recall"].iloc[0],
        "faithfulness": results_df["faithfulness"].iloc[0],
    }

    async def get_bleu_rouge_scores():
        sample = SingleTurnSample(response=answer, reference=true_ground_truth)
        
        bleu_scorer = BleuScore()
        rouge_scorer = RougeScore()

        # Smoothing function tanımla
        smoothing_function = SmoothingFunction()

        # BLEU ve ROUGE skorlarını hesapla
        bleu_score = sentence_bleu(
            [true_ground_truth.split()],
            answer.split(),
            smoothing_function=smoothing_function.method1  # Smoothing kullan
        )
        rouge_score = await rouge_scorer.single_turn_ascore(sample)

        return bleu_score, rouge_score

    bleu_score, rouge_score = asyncio.run(get_bleu_rouge_scores())
    
    results["bleu_score"] = bleu_score
    results["rouge_score"] = rouge_score

    return results, dataset