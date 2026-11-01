import os
import tempfile

# Isolate tests from any real .env or data directory.
os.environ.setdefault("PA_DATA_DIR", tempfile.mkdtemp(prefix="pa-test-"))

PAPER = """<!-- page:1 -->
# Sparse Mixture Routers for Small Models

Ada Author, Ben Builder. University of Somewhere. We study expert routing in small language models
and show that load balancing matters for both quality and stability across many training runs.

## Abstract
We study expert routing in small language models and show load balancing matters.

## 1 Introduction
Mixture of experts scales capacity without scaling compute per token.

<!-- page:2 -->
## 2 Method
### 2.1 Router
The router is a linear layer with top-2 gating and an auxiliary load balancing loss.

Table 1: Hyperparameters
| param | value |
| --- | --- |
| experts | 8 |
| top-k | 2 |

<!-- page:3 -->
### 2.2 Training
We train for 100k steps with AdamW and cosine decay. The learning rate is 3e-4.

Figure 2: Expert utilization over training.

<!-- page:4 -->
## 3 Results
Load balancing improves perplexity by 4 percent and stabilises expert utilisation.

## Acknowledgements
We thank our funders.
"""

PAPER_B = """<!-- page:1 -->
# Retrieval Heads in Long Context Transformers

## 1 Method
We probe attention heads for retrieval behaviour using needle tests. Optimizer is Lion with lr 1e-4.

<!-- page:2 -->
## 2 Results
Retrieval heads are sparse and universal across model families.
"""

META = {"title": "Sparse Mixture Routers for Small Models", "authors": ["A. Author"], "year": 2026,
        "doi": "not-a-doi", "arxiv_id": "2601.01234"}
META_B = {"title": "Retrieval Heads in Long Context Transformers", "authors": ["C. Author"], "year": 2025}
