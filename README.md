# LLM QA Lab

The goal of this project is to provide a user-friendly framework for testing Question-Answering (QA) with Large Language Models (LLMs). This need arose with the use of Retrieval-Augmented Generation (RAG), where traditional training, validation, and testing datasets are not always available. Instead, users can provide expected answers based on custom inputs, enabling flexible evaluation without predefined datasets.


## Roadmap

### 0.1

* Proof of concept: Gather input and expected output from a SQLite database and retrieve the generated answer of a LLM.

### 0.2

* Add the evaluation metrics from the article: [A Practical Guide to Evaluating Large Language Models (LLM)](https://medium.com/@thomas.zilliox/a-practical-guide-to-evaluating-large-language-models-llm-4882fb22892f)
  * BLEU
  * ROUGE
  * METEOR
  * Edit Distance (Levenshtein Distance)
  * Cosine Similarity
  * BERTScore
  * LLM as a judge
* Generate a report with all the metrics and the option to generate plots
* Option to save the generated answers as well

### 0.3

* Support from more sources:
  * csv
  * json
  * PostgreSQL (sqlalchemy ?)

### 0.4

* Optimization of the generation of the answer:
  * bulk mode ?
* Ensure compatibility with API calls
  * Prepare .env for API KEYS