"""
validator.py
=============================================================================
Validation and Sanitization Engine for Smart Resume Matching System.
Ensures strict input hygiene, regex email validation, and ML score bounds.
=============================================================================
"""

import re
import html
from typing import Any
from config import logger


class ValidationError(ValueError):
    """Custom exception raised when input validation fails."""
    pass


class InvalidScoreError(ValueError):
    """Custom exception raised when similarity score is outside valid bounds."""
    pass


class InputValidator:
    """Production input validation and sanitization utility class."""

    # RFC 5322 compliant simplified email regex pattern
    EMAIL_REGEX = re.compile(
        r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    )

    # HTML tag strip pattern
    HTML_TAG_REGEX = re.compile(r"<[^>]+>")

    # Multiple whitespace normalization pattern
    WHITESPACE_REGEX = re.compile(r"\s+")

    @classmethod
    def sanitize_text(cls, text: Any, field_name: str = "Input Text", min_length: int = 10) -> str:
        """
        Sanitizes input text by:
        1. Checking type is string.
        2. Unescaping HTML entities.
        3. Stripping HTML tags.
        4. Removing non-printable / control characters while preserving punctuation.
        5. Normalizing whitespace (replacing multiple tabs/spaces/newlines with single space).
        6. Verifying minimum content length.
        """
        if not isinstance(text, str):
            logger.error("Validation failed: %s must be a string, got %s", field_name, type(text).__name__)
            raise ValidationError(f"{field_name} must be a valid string, got {type(text).__name__}.")

        cleaned = html.unescape(text)
        cleaned = cls.HTML_TAG_REGEX.sub(" ", cleaned)
        # Remove control characters except standard whitespace
        cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch in ("\n", "\r", "\t", " "))
        cleaned = cls.WHITESPACE_REGEX.sub(" ", cleaned).strip()

        if len(cleaned) < min_length:
            logger.error("Validation failed: %s content too short (%d chars, minimum %d)", field_name, len(cleaned), min_length)
            raise ValidationError(f"{field_name} must contain at least {min_length} characters of meaningful text.")

        return cleaned

    @classmethod
    def validate_email(cls, email: Any) -> str:
        """
        Validates email format strictly against RFC regex patterns.
        """
        if not isinstance(email, str):
            logger.error("Validation failed: Email must be a string")
            raise ValidationError(f"Email must be a string, got {type(email).__name__}.")

        email_clean = email.strip().lower()

        if not cls.EMAIL_REGEX.match(email_clean):
            logger.error("Validation failed: Invalid email address syntax '%s'", email)
            raise ValidationError(f"Invalid email address: '{email}'. Must follow standard user@domain.tld format.")

        return email_clean

    @classmethod
    def validate_similarity_score(cls, score: Any) -> float:
        """
        Validates that a similarity score is a valid float within [0.0, 1.0].
        """
        try:
            val = float(score)
        except (ValueError, TypeError) as e:
            logger.error("Score validation failed: Score '%s' cannot be converted to float", score)
            raise InvalidScoreError(f"Similarity score must be a numerical value, got {type(score).__name__}: {e}")

        if val < 0.0 or val > 1.0:
            logger.error("Score validation failed: Score %.4f is outside valid range [0.0, 1.0]", val)
            raise InvalidScoreError(f"Similarity score {val:.4f} is out of bounds. Must be within [0.0, 1.0].")

        return val

    @classmethod
    def validate_skills_list(cls, skills: Any, field_name: str = "Skills") -> list:
        """
        Ensures skills is a list of non-empty strings.
        """
        if not isinstance(skills, (list, tuple, set)):
            logger.error("Validation failed: %s must be an iterable of strings", field_name)
            raise ValidationError(f"{field_name} must be a list of skill strings.")

        sanitized_skills = []
        for s in skills:
            if isinstance(s, str) and s.strip():
                sanitized_skills.append(s.strip().lower())

        return sorted(list(set(sanitized_skills)))
