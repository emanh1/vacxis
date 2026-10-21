"""Generation package for VacxiS Capstone: Grounded Medical Answer Synthesis."""
from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.prompts import (
    DEFAULT_MEDICAL_SYSTEM_PROMPT,
    build_messages,
    format_context,
)
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer, GenerationConfig

__all__ = [
    "MedicalGenerator",
    "VacxisRAGPipeline",
    "GenerationConfig",
    "GeneratedAnswer",
    "DEFAULT_MEDICAL_SYSTEM_PROMPT",
    "format_context",
    "build_messages",
]
