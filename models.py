"""
models.py
=============================================================================
Data Model Definitions for Smart Resume Matching System.
Strongly typed dataclasses representing Resume, JobDescription, and MatchResult.
=============================================================================
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class Resume:
    """Represents a validated candidate resume entity."""
    id: str
    candidate_name: str
    email: str
    raw_text: str
    cleaned_text: str
    skills: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the resume instance to a dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return f"Resume(id={self.id}, name='{self.candidate_name}', email='{self.email}', skills_count={len(self.skills)})"


@dataclass
class JobDescription:
    """Represents a job vacancy/posting entity."""
    id: str
    title: str
    department: str
    raw_text: str
    cleaned_text: str
    required_skills: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the job description instance to a dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return f"JobDescription(id={self.id}, title='{self.title}', department='{self.department}', req_skills_count={len(self.required_skills)})"


@dataclass
class MatchResult:
    """Represents the computed ML similarity scoring and skill gap audit."""
    candidate_id: str
    job_id: str
    match_score: float                # Raw cosine similarity in [0.0, 1.0]
    match_percentage: float           # Percentage representation [0.0, 100.0]
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    recommendation: str = "Under Review"
    id: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the match result instance to a dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"MatchResult(candidate={self.candidate_id}, job={self.job_id}, "
            f"score={self.match_percentage:.2f}%, tier='{self.recommendation}')"
        )
