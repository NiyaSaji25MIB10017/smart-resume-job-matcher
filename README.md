# Smart Resume Matching System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Storage-SQLite3-green.svg)](https://www.sqlite.org/)
[![VITyarthi](https://img.shields.io/badge/Submission-VITyarthi%20AI%2FML-purple.svg)](#)

A production-grade, modular AI/ML system built to evaluate, score, and rank resumes against target job descriptions using **TF-IDF Vectorization**, **Cosine Similarity**, and **Automated Skill Gap Auditing**.

---

## 1. Architectural Overview

```
                      +-----------------------------+
                      | Raw Resumes & Job Postings  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   validator.py (Sanitize)   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    models.py (Dataclasses)  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  processor.py (ML Engine)   |
                      |  - TF-IDF Vectorizer        |
                      |  - Cosine Similarity        |
                      |  - Skill Gap Analyzer       |
                      +-----------------------------+
                                     |
                      +--------------+--------------+
                      |                             |
                      v                             v
       +----------------------------+  +--------------------------+
       |   database.py (SQLite)     |  |   config.py (app.log)    |
       |   - candidates             |  |   - Audit Trail          |
       |   - job_postings           |  |   - Telemetry            |
       |   - scoring_records        |  |                          |
       +----------------------------+  +--------------------------+
```

### Mathematical Foundations

1. **TF-IDF Calculation**:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   - Sublinear term frequency scaling is applied: $\text{TF}(t, d) = 1 + \log(\text{count}(t, d))$ for $\text{count} > 0$.
   - Unigram and Bigram combinations $(1, 2)$ capture phrases such as `machine learning`, `deep learning`, `ci/cd`.

2. **Cosine Similarity**:
   $$\text{Similarity}(\mathbf{r}, \mathbf{j}) = \frac{\mathbf{r} \cdot \mathbf{j}}{\|\mathbf{r}\|_2 \|\mathbf{j}\|_2} = \frac{\sum_{i=1}^n r_i j_i}{\sqrt{\sum_{i=1}^n r_i^2} \sqrt{\sum_{i=1}^n j_i^2}}$$
   - Produces an exact geometric similarity score bounded strictly in $[0.0, 1.0]$.
   - Converted to a percentage: $\text{Score \%} = \text{Similarity} \times 100$.

3. **Skill Gap Set Theory**:
   $$\text{Matched Skills} = S_{\text{required}} \cap S_{\text{candidate}}$$
   $$\text{Missing Skills} = S_{\text{required}} \setminus S_{\text{candidate}}$$

---

## 2. Project Directory Structure

```
vityarthi_resume_ml/
│
├── config.py             # Centralized configurations, ML thresholds, and dual logging setup
├── validator.py          # Strict text sanitization, RFC email validation, and score range checks
├── models.py             # Strongly typed dataclasses (Resume, JobDescription, MatchResult)
├── database.py           # SQLite manager, relational schema, and query APIs (vityarthi_ml.db)
├── processor.py          # Core ML engine (TF-IDF, Cosine Similarity, Skill Gap Extraction)
├── main.py               # End-to-end execution pipeline with sample data and reporting
│
├── requirements.txt      # Project Python dependencies (scikit-learn, numpy)
├── statement.md          # Official VITyarthi AI/ML Problem Statement & Scope document
├── README.md             # Project documentation, execution, and architecture guide
│
├── vityarthi_ml.db       # Generated SQLite database (persisted candidates & scoring history)
└── app.log               # Production audit log with timestamped event logging
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10 or higher installed.
- Git (optional, for version control).

### Step 1: Clone or Navigate to Project Directory
```bash
cd vityarthi_resume_ml
```

### Step 2: (Recommended) Create a Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Running the System

Run the main pipeline in the terminal:

```bash
python main.py
```

### Expected Output
The system executes the full end-to-end pipeline:
1. Initializes the SQLite relational schema in `vityarthi_ml.db`.
2. Validates and ingests job postings and candidate resumes.
3. Calculates TF-IDF vectors and computes pairwise cosine similarity scores.
4. Identifies matched skills and missing skill gaps.
5. Persists evaluation records to SQLite.
6. Renders a formatted terminal dashboard and writes logs to `app.log`.

Sample Console View:
```
================================================================================
                    SMART RESUME MATCHING SYSTEM
                  Production AI/ML Pipeline Execution
================================================================================

[+] Ingesting & Validating Job Postings...
  -> Ingested: [JOB-101] Senior AI / Machine Learning Engineer (7 required skills)
  -> Ingested: [JOB-102] Full-Stack Cloud Engineer (7 required skills)

[+] Ingesting & Validating Candidate Resumes...
  -> Ingested: [RES-001] Dr. Sarah Chen <sarah.chen@vitstudent.ac.in>
  -> Ingested: [RES-002] Alex Mercer <alex.mercer@techfirm.io>
  -> Ingested: [RES-003] Rohan Sharma <rohan.sharma@vityarthi.edu>

[+] Running ML Engine (TF-IDF Vectorizer + Cosine Similarity)...

==========================================================================================
CANDIDATE          | TARGET ROLE                  | SCORE   | TIER RECOMMENDATION         
------------------------------------------------------------------------------------------
Dr. Sarah Chen     | Senior AI / Machine Learning |  74.82% | Strongly Recommended (Tier 1)
Alex Mercer        | Senior AI / Machine Learning |  31.54% | Under Review (Partial Match)
Rohan Sharma       | Senior AI / Machine Learning |  28.91% | Not Recommended (High Gap)  
Dr. Sarah Chen     | Full-Stack Cloud Engineer    |  33.20% | Under Review (Partial Match)
Alex Mercer        | Full-Stack Cloud Engineer    |  78.15% | Strongly Recommended (Tier 1)
Rohan Sharma       | Full-Stack Cloud Engineer    |  21.40% | Not Recommended (High Gap)  
==========================================================================================
```

---

## 5. Database Schema & Storage (`vityarthi_ml.db`)

The SQLite database enforces relational consistency and fast indexed lookups:

- **`candidates`**: `id` (PK), `name`, `email` (UNIQUE), `skills`, `raw_text`, `cleaned_text`, `created_at`
- **`job_postings`**: `id` (PK), `title`, `department`, `required_skills`, `raw_text`, `cleaned_text`, `created_at`
- **`scoring_records`**: `id` (PK AUTOINCREMENT), `candidate_id` (FK), `job_id` (FK), `match_score`, `match_percentage`, `matched_skills`, `missing_skills`, `recommendation`, `timestamp`

---

## 6. Audit Logging (`app.log`)

Every operational event is recorded with strict timestamps and log levels (`INFO`, `WARNING`, `ERROR`):
```log
[2026-09-18 09:16:05] [INFO] [ResumeMatcher:init_db:42] Initializing SQLite database at: .../vityarthi_ml.db
[2026-09-18 09:16:05] [INFO] [ResumeMatcher:insert_candidate:95] Persisted candidate profile: Dr. Sarah Chen (RES-001)
[2026-09-18 09:16:05] [INFO] [ResumeMatcher:evaluate_match:142] Evaluation Complete -> Score: 74.82% | Matched: 6 | Missing: 1 | Decision: Strongly Recommended (Tier 1)
```

---

## 7. Submission Checklist

- [x] 6 modular files (`config.py`, `validator.py`, `models.py`, `database.py`, `processor.py`, `main.py`)
- [x] Scikit-learn TF-IDF & Cosine Similarity ML Engine
- [x] Input sanitization and email validation
- [x] SQLite persistence (`vityarthi_ml.db`)
- [x] Dual logging to console and `app.log`
- [x] Detailed Problem Statement document (`statement.md`)
- [x] Setup and testing documentation (`README.md`)
- [x] Zero runtime errors
