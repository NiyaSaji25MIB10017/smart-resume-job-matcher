"""
processor.py
=============================================================================
Core Machine Learning Engine for Smart Resume Matching System.
Implements TF-IDF Vectorization, Cosine Similarity scoring, technical skill
keyword extraction, and gap analysis using scikit-learn and numpy.
=============================================================================
"""

import re
from typing import List, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CONFIG, logger
from validator import InputValidator
from models import Resume, JobDescription, MatchResult


# Canonical Technical Skills Taxonomy for keyword extraction and matching
TECH_SKILLS_TAXONOMY = {
    # Programming Languages
    "python", "java", "c++", "c#", "golang", "rust", "javascript", "typescript",
    "r", "scala", "sql", "bash", "shell", "html", "css",
    # ML / AI / Data Science
    "machine learning", "deep learning", "nlp", "computer vision", "pytorch",
    "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "matplotlib",
    "seaborn", "huggingface", "llm", "transformers", "xgboost", "lightgbm",
    "data science", "statistics", "feature engineering", "mlops", "langchain",
    # Cloud & DevOps
    "docker", "kubernetes", "aws", "gcp", "azure", "ci/cd", "git", "github",
    "terraform", "ansible", "linux", "jenkins",
    # Backend & Web
    "fastapi", "flask", "django", "node.js", "express", "react", "next.js",
    "rest api", "graphql", "microservices", "redis", "postgresql", "mysql",
    "mongodb", "sqlite", "kafka", "rabbitmq"
}


class ResumeMatcher:
    """Production ML processor for resume to job description matching."""

    def __init__(self):
        self.vectorizer_config = CONFIG.VECTORIZER
        self.thresholds = CONFIG.ML
        logger.info(
            "Initialized ResumeMatcher ML Engine with N-gram=%s, MaxFeatures=%d",
            self.vectorizer_config.NGRAM_RANGE,
            self.vectorizer_config.MAX_FEATURES
        )

    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Scans normalized text against the technical skill taxonomy to extract
        present skill keywords (supports single words and multi-word phrases).
        """
        text_lower = text.lower()
        extracted: Set[str] = set()

        for skill in TECH_SKILLS_TAXONOMY:
            # Word boundary regex to avoid partial substring false positives
            # e.g., 'r' shouldn't match 'for', 'c' shouldn't match 'react'
            pattern = r"(?<![a-zA-Z0-9_\-\./])" + re.escape(skill) + r"(?![a-zA-Z0-9_\-\./])"
            if re.search(pattern, text_lower):
                extracted.add(skill)

        return sorted(list(extracted))

    def compute_similarity(self, resume_text: str, job_text: str) -> float:
        """
        Computes cosine similarity between TF-IDF representations of resume and job description.
        Returns a float strictly bounded in [0.0, 1.0].
        """
        # Create TF-IDF vectorizer instance
        vectorizer = TfidfVectorizer(
            ngram_range=self.vectorizer_config.NGRAM_RANGE,
            max_features=self.vectorizer_config.MAX_FEATURES,
            stop_words=self.vectorizer_config.STOP_WORDS,
            sublinear_tf=self.vectorizer_config.SUBLINEAR_TF,
            norm=self.vectorizer_config.NORM
        )

        # Fit and transform corpus (pair of texts)
        corpus = [resume_text, job_text]
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Compute cosine similarity between resume (idx 0) and job (idx 1)
        sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        raw_score = float(sim_matrix[0][0])

        # Clip numerical precision anomalies
        clipped_score = max(0.0, min(1.0, raw_score))

        # Validate score bounds
        validated_score = InputValidator.validate_similarity_score(clipped_score)
        return validated_score

    def analyze_skill_gaps(
        self,
        candidate_skills: List[str],
        job_required_skills: List[str],
        resume_text: str
    ) -> Tuple[List[str], List[str]]:
        """
        Performs set-based skill gap analysis:
        - Finds matched skills present in candidate's explicit skills or extracted from resume text.
        - Identifies missing skills required by the job but absent from candidate's profile.
        """
        # Combine explicit candidate skills with any discovered in resume text
        discovered_skills = set(self.extract_skills_from_text(resume_text))
        all_candidate_skills = set(candidate_skills).union(discovered_skills)

        # Normalize required skills
        required_set = set(s.lower().strip() for s in job_required_skills)

        # Intersect for matches, difference for gaps
        matched = sorted(list(required_set.intersection(all_candidate_skills)))
        missing = sorted(list(required_set.difference(all_candidate_skills)))

        return matched, missing

    def determine_recommendation(self, match_percentage: float, missing_skills_count: int) -> str:
        """
        Categorizes applicant based on cosine similarity thresholds and skill gap count.
        """
        score_ratio = match_percentage / 100.0

        if score_ratio >= self.thresholds.STRONG_MATCH:
            if missing_skills_count == 0:
                return "Strongly Recommended (Elite Match)"
            return "Strongly Recommended (Tier 1)"
        elif score_ratio >= self.thresholds.QUALIFIED_MATCH:
            return "Recommended (Qualified - Tier 2)"
        elif score_ratio >= self.thresholds.PARTIAL_MATCH:
            return "Under Review (Partial Match - Tier 3)"
        else:
            return "Not Recommended (High Skill Gap - Tier 4)"

    def evaluate_match(self, resume: Resume, job: JobDescription) -> MatchResult:
        """
        Executes end-to-end ML matching pipeline for a candidate and job description:
        1. Validates text inputs.
        2. Computes TF-IDF Cosine Similarity score.
        3. Extracts skill intersections and gap analysis.
        4. Classifies into recommendation tiers.
        5. Returns structured MatchResult entity.
        """
        logger.info("Evaluating Candidate '%s' against Job '%s'...", resume.candidate_name, job.title)

        # 1. Compute similarity score
        score = self.compute_similarity(resume.cleaned_text, job.cleaned_text)
        percentage = round(score * 100.0, 2)

        # 2. Analyze skill gaps
        matched_skills, missing_skills = self.analyze_skill_gaps(
            candidate_skills=resume.skills,
            job_required_skills=job.required_skills,
            resume_text=resume.cleaned_text
        )

        # 3. Determine recommendation tier
        recommendation = self.determine_recommendation(percentage, len(missing_skills))

        # 4. Construct MatchResult
        result = MatchResult(
            candidate_id=resume.id,
            job_id=job.id,
            match_score=score,
            match_percentage=percentage,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            recommendation=recommendation
        )

        logger.info(
            "Evaluation Complete -> Score: %.2f%% | Matched: %d | Missing: %d | Decision: %s",
            percentage, len(matched_skills), len(missing_skills), recommendation
        )
        return result
