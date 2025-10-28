import sqlite3
import torch
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


class QABenchmark:
    def __init__(self,db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect('fake_db.db')
        self.cursor = self.conn.cursor()


    def __del__(self):
        self.conn.close()


    def stream_batches(self, batch_size=20):
        batch = []
        for row in self.conn.execute("SELECT question, answer FROM qa"):
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch



    def evaluate_batch(self, batch, tokenizer, model, device, max_new_tokens=50):
        results = []
        for question, expected_answer in batch:
            prompt = (
                "You must answer in one or two concise sentences. "
                "Do not explain or add extra details.\n"
                f"Question: {question}\nAnswer:"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated.split("Answer:")[-1].strip()

            # print(generated_answer.split())
            # print(expected_answer.split())

            bleu_score  = sentence_bleu(hypothesis=generated_answer.split(), references=[expected_answer.split()])
            rouge_score = scorer.score(expected_answer, generated_answer)
            single_score = rouge_score["rougeL"].fmeasure


            evaluation = {
                "question": question,
                "answer": generated_answer,
                "expected_answer": expected_answer,
                "bleu_score": bleu_score,
                "rouge_score": single_score
            }
            results.append(evaluation)
        return results

    def evaluate_all(self, tokenizer, model, device, batch_size=20):
        all_results = []
        for batch in self.stream_batches(batch_size=batch_size):
            batch_results = self.evaluate_batch(batch, tokenizer, model, device)
            all_results.extend(batch_results)
        return all_results

