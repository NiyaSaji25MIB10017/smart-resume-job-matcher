# Smart Resume Matching & Skill Gap Engine
Student Name: Niya Saji

Registration No: 25MIB10017

Program: VITyarthi Build Your Own Project (AI/ML)

This is an AI/ML project built to automatically compare candidate resumes against job descriptions. It uses TF-IDF text vectorization and cosine similarity to calculate match percentages and highlight missing skills.

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
How the Matching Works
1. **Text Cleaning**: validator.py cleans input text and verifies candidate emails.
2. **ML Vectorization**: processor.py converts resume and job description text into TF-IDF vectors using scikit-learn.
3. **Similarity Score**: Calculates the cosine similarity between vectors to give a match score percentage.
4. **Database Storage**: Saves evaluation results directly into SQLite (vityarthi_ml.db).   



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

- [x] Modular Python code split into 6 files
- [x] Machine Learning logic using Scikit-Learn (TF-IDF & Cosine Similarity)   -
- [x] Input sanitization and email check
- [x] SQLite database storage for results
- [x] Error handling and event logging in app.log
- [x] Complete README and project statement   
