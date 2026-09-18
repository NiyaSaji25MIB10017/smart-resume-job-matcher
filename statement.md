# Smart Resume Matching System — Project Statement

**Academic & Technical Submission Document**  
**Course/Platform**: VITyarthi AI/ML Track  
**Project Title**: Smart Resume Matching System  
**Author / Engineering Role**: Principal Machine Learning Engineer  

---

## 1. Problem Statement

In contemporary recruitment workflows, talent acquisition teams and campus placement cells face an overwhelming volume of candidate applications. Traditional Applicant Tracking Systems (ATS) and manual screening workflows suffer from severe operational and algorithmic bottlenecks:

1. **Keyword-Stuffing & Fragile Regex Matching**: Traditional boolean ATS systems rely on exact keyword matches. Candidates who use synonyms, related terminology, or modern phrasing are frequently disqualified, while candidates who artificially pad resumes with hidden keywords pass initial filters without true contextual qualifications.
2. **High Latency & Human Fatigue**: Manual evaluation of hundreds of technical resumes per opening takes dozens of hours, introducing cognitive fatigue, inconsistency, and unconscious bias into the initial screening stage.
3. **Absence of Actionable Skill Gap Auditing**: Existing screeners provide binary pass/fail verdicts or arbitrary scores without detailing *why* a candidate fell short, what required skills were missing, or what competencies directly matched the job description.
4. **Lack of Enterprise-Grade Data Governance**: Many prototype ML tools fail to provide structured relational data logging, input sanitization against malicious payloads, or audit trails for compliance.

---

## 2. Proposed AI/ML Solution

The **Smart Resume Matching System** is an enterprise-ready, modular Natural Language Processing (NLP) system designed to automate resume evaluation objectively.

The solution leverages:
- **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorization with sublinear scaling and bi-gram feature extraction to measure term relevance while discounting universally common stop words.
- **Cosine Similarity** metric in high-dimensional vector space to evaluate semantic proximity between applicant resumes and target job specifications, producing a bounded `[0.0, 1.0]` (0% to 100%) match percentage.
- **Set-Theoretic Skill Gap Engine** that compares applicant competencies against the job's mandatory skill taxonomy, isolating both *matched proficiencies* and *critical missing skills*.
- **Multi-Tier Classification Hierarchy**:
  - **Tier 1 (≥ 70%)**: Strongly Recommended (Elite Match)
  - **Tier 2 (50% – 69%)**: Recommended (Qualified Match)
  - **Tier 3 (30% – 49%)**: Under Review (Partial Match)
  - **Tier 4 (< 30%)**: Not Recommended (High Skill Gap)

---

## 3. Project Scope

### In Scope
- **Input Sanitization & Validation**: HTML tag stripping, non-printable character removal, email RFC-5322 compliance checking, and similarity score range assertion.
- **Feature Extraction & NLP Processing**: Scikit-learn TF-IDF n-gram vectorizer, cosine similarity calculation, and automated domain taxonomy skill extraction.
- **Skill Gap Auditing**: Explicit keyword intersection and missing requirement detection.
- **Relational Persistence (SQLite)**: Schema with `candidates`, `job_postings`, and `scoring_records` linked by foreign keys and optimized with indexes.
- **Audit Logging**: Dual-output logging (stdout and `app.log`) capturing operational telemetry, pipeline execution, and model decisions.
- **Modular Production Codebase**: Strict separation of concerns across 6 modules (`config.py`, `validator.py`, `models.py`, `database.py`, `processor.py`, `main.py`).

### Out of Scope (Future Roadmap)
- Optical Character Recognition (OCR) for scanned PDFs (current version handles digital text/resumes).
- Distributed cluster processing with Apache Spark.
- Deep Neural Bi-Encoder embeddings (e.g., Sentence-BERT / RoBERTa) which can be plugged into the modular `processor.py` engine in subsequent releases.

---

## 4. Target Users

| User Persona | Primary Use Case & Value Proposition |
| :--- | :--- |
| **Campus Placement Cells (VITyarthi)** | Automatically rank hundreds of student resumes against corporate job descriptions to recommend best-fit candidates for on-campus drives. |
| **Technical Recruiters & HR Leads** | Eliminate manual triage time by instantly filtering applicants into clear recommendation tiers with highlighted skill gaps. |
| **Hiring Managers** | Review shortlist reports showing exact skill coverage before scheduling technical interviews. |
| **Student Applicants & Job Seekers** | Gain clear visibility into missing technical keywords to tailor upskilling initiatives. |

---

## 5. High-Level System Features

```
+-----------------------------------------------------------------------------+
|                      SMART RESUME MATCHING SYSTEM                           |
+-----------------------------------------------------------------------------+
   |
   +---> 1. Data Ingestion & Sanitization Layer (validator.py)
   |        - HTML / Control character stripping
   |        - Strict RFC Email verification
   |        - Bounded range checks
   |
   +---> 2. Strongly-Typed Data Modeling (models.py)
   |        - Dataclasses: Resume, JobDescription, MatchResult
   |        - Serialization and auditing hooks
   |
   +---> 3. Machine Learning Engine (processor.py)
   |        - TF-IDF Vectorizer with Sublinear TF & Bi-grams
   |        - Cosine Similarity Metric
   |        - Technical Skill Taxonomy Matching & Gap Analysis
   |        - Recommendation Tier Decision Logic
   |
   +---> 4. Relational Storage & Indexing (database.py)
   |        - SQLite database: vityarthi_ml.db
   |        - Tables: candidates, job_postings, scoring_records
   |        - Relational Integrity (Foreign Keys & Cascading Deletes)
   |
   +---> 5. Centralized Configuration & Telemetry (config.py)
   |        - Configurable hyperparameters & scoring thresholds
   |        - Dual-output logging to console and app.log
   |
   +---> 6. Orchestration & Presentation Layer (main.py)
            - End-to-end execution pipeline
            - Formatted terminal dashboard and ranking breakdown
```

---

## 6. Technical Stack Summary

- **Core Language**: Python 3.10+
- **Machine Learning & NLP**: `scikit-learn`, `numpy`
- **Database**: SQLite3 (`vityarthi_ml.db`)
- **Data Modeling**: Python Standard Library `dataclasses`, `typing`, `json`
- **Logging**: Python Standard Library `logging` (`app.log`)
- **Validation**: Python Standard Library `re`, `html`
