## Generation evaluation (RAGAS)

Generator: gpt-4.1-mini | judge: gpt-4.1 | embeddings: text-embedding-3-small | retrieval: hybrid, top_k=8

| Query | Faithfulness | Answer relevancy | Context precision | Context recall | Citations | Verified h | Reference h |
|---|---|---|---|---|---|---|---|
| q1-fashion-store | 0.92 | 0.67 | 1.00 | 0.57 | 4/4 grounded | 226 | 216 |
| q2-marketplace | 1.00 | 0.46 | 1.00 | 0.62 | 0/4 grounded, 4 unverified_evidence | 0 | 830 |
| q3-telehealth | 1.00 | 0.65 | 0.33 | 1.00 | 5/5 grounded | 790 | 850 |
| q4-predictive-maintenance | 1.00 | 0.77 | 0.83 | 0.80 | 3/3 grounded | 500 | 500 |
| q5-warehouse-logistics | 0.62 | 0.74 | 0.00 | 1.00 | 0/3 grounded, 3 unverified_evidence | 0 | 640 |
| **Average** | **0.91** | **0.66** | **0.63** | **0.80** | | | |