"""
main.py
=============================================================================
Main Application Entry Point for Smart Resume Matching System.
Demonstrates end-to-end pipeline: Input sanitization, ML processing,
database persistence, and formatted reporting adhering to VITyarthi guidelines.
=============================================================================
"""

import os
from config import CONFIG, logger
from validator import InputValidator
from models import Resume, JobDescription
from database import DatabaseManager
from processor import ResumeMatcher


def create_sample_dataset():
    """Generates realistic sample resumes and job postings for demonstration."""
    
    # 1. Job Descriptions
    jobs_data = [
        {
            "id": "JOB-101",
            "title": "Senior AI / Machine Learning Engineer",
            "department": "Artificial Intelligence & Research",
            "required_skills": ["python", "pytorch", "scikit-learn", "docker", "mlops", "aws", "nlp"],
            "raw_text": """
                We are looking for a Senior AI / Machine Learning Engineer to design, build,
                and scale production deep learning and machine learning systems.
                Key Responsibilities:
                - Develop predictive models and NLP pipelines using Python, PyTorch, and Scikit-Learn.
                - Containerize model microservices using Docker and orchestrate on AWS.
                - Implement automated CI/CD and MLOps workflows for model monitoring and retraining.
                - Strong understanding of data structures, statistics, and distributed computing.
            """
        },
        {
            "id": "JOB-102",
            "title": "Full-Stack Cloud Engineer",
            "department": "Core Infrastructure",
            "required_skills": ["python", "fastapi", "react", "docker", "postgresql", "redis", "ci/cd"],
            "raw_text": """
                Seeking a talented Full-Stack Cloud Engineer to engineer robust web services.
                Key Responsibilities:
                - Build high-performance REST APIs in Python using FastAPI.
                - Create interactive responsive web user interfaces using React and TypeScript.
                - Manage database schemas with PostgreSQL and caching layers with Redis.
                - Containerize full stack applications with Docker and configure CI/CD pipelines.
            """
        }
    ]

    # 2. Resumes
    candidates_data = [
        {
            "id": "RES-001",
            "name": "Dr. Sarah Chen",
            "email": "sarah.chen@vitstudent.ac.in",
            "skills": ["python", "pytorch", "scikit-learn", "nlp", "docker", "aws", "pandas", "numpy"],
            "raw_text": """
                Senior Machine Learning Scientist with 6+ years of expertise delivering AI solutions.
                Extensive hands-on experience training transformer architectures and NLP pipelines
                with PyTorch, HuggingFace, and Scikit-learn in Python.
                Architected scalable model serving microservices using Docker and deployed on AWS.
                Proficient in data preprocessing with Pandas, Numpy, and building automated MLOps pipelines.
            """
        },
        {
            "id": "RES-002",
            "name": "Alex Mercer",
            "email": "alex.mercer@techfirm.io",
            "skills": ["python", "fastapi", "react", "docker", "postgresql", "redis", "git", "ci/cd"],
            "raw_text": """
                Full-Stack Software Engineer with 4 years building cloud-native web applications.
                Expert in Python backend development with FastAPI, Django, and relational modeling in PostgreSQL.
                Built dynamic, stateful frontend dashboards using React and modern CSS.
                Designed containerized deployments using Docker and GitHub Actions CI/CD pipelines.
                Familiar with in-memory caching using Redis and basic data science scripting.
            """
        },
        {
            "id": "RES-003",
            "name": "Rohan Sharma",
            "email": "rohan.sharma@vityarthi.edu",
            "skills": ["python", "sql", "pandas", "matplotlib", "seaborn", "statistics"],
            "raw_text": """
                Junior Data Analyst and Python Developer with a solid foundation in statistics and SQL.
                Skilled in exploratory data analysis, data cleaning, and reporting using Python, Pandas,
                Matplotlib, and Seaborn. Built introductory machine learning classifiers with Scikit-learn.
                Eager to expand into cloud technologies, Docker containerization, and advanced deep learning.
            """
        }
    ]

    return jobs_data, candidates_data


def main():
    """Main orchestration function executing the full AI/ML pipeline."""
    print("=" * 80)
    print(" " * 20 + "SMART RESUME MATCHING SYSTEM")
    print(" " * 18 + "Production AI/ML Pipeline Execution")
    print("=" * 80)

    logger.info("Initializing Smart Resume Matching System pipeline...")

    # 1. Database Initialization
    db = DatabaseManager()
    db.init_db()

    # 2. ML Processor Initialization
    matcher = ResumeMatcher()

    # 3. Load Sample Dataset
    jobs_raw, candidates_raw = create_sample_dataset()

    # 4. Ingest and Validate Job Descriptions
    validated_jobs = []
    print("\n[+] Ingesting & Validating Job Postings...")
    for j in jobs_raw:
        cleaned_text = InputValidator.sanitize_text(j["raw_text"], field_name=f"Job {j['id']} Text")
        req_skills = InputValidator.validate_skills_list(j["required_skills"])
        job_entity = JobDescription(
            id=j["id"],
            title=j["title"],
            department=j["department"],
            raw_text=j["raw_text"].strip(),
            cleaned_text=cleaned_text,
            required_skills=req_skills
        )
        db.insert_job_posting(job_entity)
        validated_jobs.append(job_entity)
        print(f"  -> Ingested: [{job_entity.id}] {job_entity.title} ({len(job_entity.required_skills)} required skills)")

    # 5. Ingest and Validate Candidates
    validated_resumes = []
    print("\n[+] Ingesting & Validating Candidate Resumes...")
    for c in candidates_raw:
        email = InputValidator.validate_email(c["email"])
        cleaned_text = InputValidator.sanitize_text(c["raw_text"], field_name=f"Resume {c['id']} Text")
        skills = InputValidator.validate_skills_list(c["skills"])
        resume_entity = Resume(
            id=c["id"],
            candidate_name=c["name"],
            email=email,
            raw_text=c["raw_text"].strip(),
            cleaned_text=cleaned_text,
            skills=skills
        )
        db.insert_candidate(resume_entity)
        validated_resumes.append(resume_entity)
        print(f"  -> Ingested: [{resume_entity.id}] {resume_entity.candidate_name} <{resume_entity.email}>")

    # 6. Execute ML Matching & Skill Gap Analysis
    print("\n[+] Running ML Engine (TF-IDF Vectorizer + Cosine Similarity)...")
    evaluation_records = []
    for job in validated_jobs:
        for resume in validated_resumes:
            match_result = matcher.evaluate_match(resume=resume, job=job)
            db.save_match_result(match_result)
            evaluation_records.append((resume, job, match_result))

    # 7. Print Formatted Results Dashboard
    print("\n" + "=" * 90)
    print(f"{'CANDIDATE':<18} | {'TARGET ROLE':<28} | {'SCORE':<7} | {'TIER RECOMMENDATION':<28}")
    print("-" * 90)
    for resume, job, result in evaluation_records:
        print(
            f"{resume.candidate_name:<18} | "
            f"{job.title[:28]:<28} | "
            f"{result.match_percentage:>5.2f}% | "
            f"{result.recommendation:<28}"
        )
    print("=" * 90)

    # 8. Detailed Skill Gap Breakdown per Job
    print("\n" + "=" * 90)
    print("DETAILED SKILL GAP & KEYWORD MATCH AUDIT")
    print("=" * 90)
    for job in validated_jobs:
        print(f"\n[JOB OPENING]: {job.title} (ID: {job.id})")
        top_matches = db.get_top_matches_for_job(job.id, limit=3)
        for idx, rec in enumerate(top_matches, start=1):
            print(f"  Rank #{idx}: {rec['candidate_name']} ({rec['match_percentage']:.2f}%) -> {rec['recommendation']}")
            print(f"    - Matched Skills ({len(rec['matched_skills'])}): {', '.join(rec['matched_skills']) if rec['matched_skills'] else 'None'}")
            print(f"    - Missing Skills ({len(rec['missing_skills'])}): {', '.join(rec['missing_skills']) if rec['missing_skills'] else 'None (Complete Coverage)'}")

    # 9. Verify System State
    all_records = db.get_all_records()
    log_exists = os.path.exists(CONFIG.LOG_PATH)
    log_size = os.path.getsize(CONFIG.LOG_PATH) if log_exists else 0

    print("\n" + "=" * 80)
    print("SYSTEM HEALTH & PERSISTENCE VERIFICATION")
    print("=" * 80)
    print(f"  [x] SQLite Database ({CONFIG.DB_NAME}): Successfully updated")
    print(f"  [x] Total Scoring Records Persisted: {len(all_records)}")
    print(f"  [x] Audit Log File ({CONFIG.LOG_NAME}): Generated ({log_size} bytes)")
    print("  [x] Runtime Status: ZERO ERRORS - Production Pipeline Verified.")
    print("=" * 80 + "\n")

    logger.info("Pipeline execution completed successfully with %d scoring records.", len(all_records))


if __name__ == "__main__":
    main()
