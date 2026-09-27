| Configuration | Precision | Recall | F1 | False alarms (tricky genuine) |
|---|---|---|---|---|
| Full system | 1.00 | 0.88 | 0.93 | 0% |
| Rules only | 1.00 | 0.75 | 0.86 | 0% |
| Similarity only | 0.73 | 1.00 | 0.84 | 50% |
| Without rules | 0.73 | 1.00 | 0.84 | 50% |
| Without similarity | 1.00 | 0.75 | 0.86 | 0% |

Recall on hidden-text variants: **100%**  
Attack success rate (simulated agent, 7 attacks that worked without protection): **100% → 0%** with PromptShield, **0%** with the action gate alone  
Median scan time: **1.1 ms** (p95 1.5 ms)  
_Test set: data/splits/test.jsonl (16 rows); similarity: tfidf; classifier: off; judge: off; generated 2026-09-26T19:00:54_
