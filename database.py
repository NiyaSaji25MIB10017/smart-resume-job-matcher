"""
database.py
=============================================================================
Database Storage Engine for Smart Resume Matching System.
Manages SQLite connection, relational schema initialization, and transactional
persistence for candidate profiles, job descriptions, and match scoring logs.
=============================================================================
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any
from config import CONFIG, logger
from models import Resume, JobDescription, MatchResult


class DatabaseManager:
    """Manages SQLite operations and relational queries for the ML system."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or CONFIG.DB_PATH

    def get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection with foreign key support."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initializes database tables and indexes if they do not exist."""
        logger.info("Initializing SQLite database at: %s", self.db_path)
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Candidates table (Resume Profiles)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    skills TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );
            """)

            # 2. Job Postings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_postings (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    department TEXT NOT NULL,
                    required_skills TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );
            """)

            # 3. Scoring Records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scoring_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    match_score REAL NOT NULL,
                    match_percentage REAL NOT NULL,
                    matched_skills TEXT NOT NULL,
                    missing_skills TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
                    FOREIGN KEY (job_id) REFERENCES job_postings(id) ON DELETE CASCADE
                );
            """)

            # Indexes for faster query lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scoring_candidate ON scoring_records (candidate_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scoring_job ON scoring_records (job_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scoring_score ON scoring_records (match_percentage DESC);")

            conn.commit()
            logger.info("Database schema initialized successfully.")

    def insert_candidate(self, resume: Resume) -> None:
        """Persists or updates a candidate resume in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            skills_json = json.dumps(resume.skills)
            cursor.execute("""
                INSERT OR REPLACE INTO candidates (id, name, email, skills, raw_text, cleaned_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                resume.id,
                resume.candidate_name,
                resume.email,
                skills_json,
                resume.raw_text,
                resume.cleaned_text,
                resume.created_at
            ))
            conn.commit()
            logger.info("Persisted candidate profile: %s (%s)", resume.candidate_name, resume.id)

    def insert_job_posting(self, job: JobDescription) -> None:
        """Persists or updates a job description in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            req_skills_json = json.dumps(job.required_skills)
            cursor.execute("""
                INSERT OR REPLACE INTO job_postings (id, title, department, required_skills, raw_text, cleaned_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                job.id,
                job.title,
                job.department,
                req_skills_json,
                job.raw_text,
                job.cleaned_text,
                job.created_at
            ))
            conn.commit()
            logger.info("Persisted job posting: %s (%s)", job.title, job.id)

    def save_match_result(self, result: MatchResult) -> int:
        """Records an evaluation match result and returns the record ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scoring_records (
                    candidate_id, job_id, match_score, match_percentage,
                    matched_skills, missing_skills, recommendation, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                result.candidate_id,
                result.job_id,
                result.match_score,
                result.match_percentage,
                json.dumps(result.matched_skills),
                json.dumps(result.missing_skills),
                result.recommendation,
                result.timestamp
            ))
            conn.commit()
            record_id = cursor.lastrowid
            result.id = record_id
            logger.info(
                "Persisted match result ID=%d: Candidate=%s to Job=%s (Score=%.2f%%, Tier='%s')",
                record_id, result.candidate_id, result.job_id, result.match_percentage, result.recommendation
            )
            return record_id

    def get_top_matches_for_job(self, job_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top scoring candidates for a specific job."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    sr.id,
                    sr.candidate_id,
                    c.name AS candidate_name,
                    c.email,
                    sr.job_id,
                    jp.title AS job_title,
                    sr.match_score,
                    sr.match_percentage,
                    sr.matched_skills,
                    sr.missing_skills,
                    sr.recommendation,
                    sr.timestamp
                FROM scoring_records sr
                JOIN candidates c ON sr.candidate_id = c.id
                JOIN job_postings jp ON sr.job_id = jp.id
                WHERE sr.job_id = ?
                ORDER BY sr.match_percentage DESC
                LIMIT ?;
            """, (job_id, limit))

            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["matched_skills"] = json.loads(item["matched_skills"])
                item["missing_skills"] = json.loads(item["missing_skills"])
                results.append(item)
            return results

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Retrieves all match scoring records with joined candidate and job titles."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    sr.id,
                    sr.candidate_id,
                    c.name AS candidate_name,
                    sr.job_id,
                    jp.title AS job_title,
                    sr.match_score,
                    sr.match_percentage,
                    sr.matched_skills,
                    sr.missing_skills,
                    sr.recommendation,
                    sr.timestamp
                FROM scoring_records sr
                JOIN candidates c ON sr.candidate_id = c.id
                JOIN job_postings jp ON sr.job_id = jp.id
                ORDER BY sr.id ASC;
            """)
            rows = cursor.fetchall()
            records = []
            for row in rows:
                item = dict(row)
                item["matched_skills"] = json.loads(item["matched_skills"])
                item["missing_skills"] = json.loads(item["missing_skills"])
                records.append(item)
            return records
