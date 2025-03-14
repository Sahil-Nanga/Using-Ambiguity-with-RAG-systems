
from new_src.pipeline import RagPipeline
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")["test"]

# Initialize pipeline
p = RagPipeline()

# Initialize dictionaries to store scores for each variation
precision_scores = {}
recall_scores = {}
f1_scores = {}

for question in dataset:
    query = question["question"]
    relevant_ids = set(question["relevant_passage_ids"])  # Ground truth passage IDs

    versions = p.ask_query(query, make_query_ambiguous=True)
    
    for version, retrieved_ids in versions.items():
        if version not in precision_scores:
            precision_scores[version] = []
            recall_scores[version] = []
            f1_scores[version] = []

        retrieved_ids = set(retrieved_ids)  # Convert to set for comparison
        intersection = retrieved_ids.intersection(relevant_ids)

        # Compute precision, recall, and F1-score
        precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(intersection) / len(relevant_ids) if relevant_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Store scores
        precision_scores[version].append(precision)
        recall_scores[version].append(recall)
        f1_scores[version].append(f1)

# Compute average precision, recall, and F1-score for each variation
for version in precision_scores.keys():
    avg_precision = sum(precision_scores[version]) / len(precision_scores[version])
    avg_recall = sum(recall_scores[version]) / len(recall_scores[version])
    avg_f1 = sum(f1_scores[version]) / len(f1_scores[version])

    print(f"Version: {version}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")
    print(f"  Average F1-Score: {avg_f1:.4f}")
    print("____________________________________________________________________")
