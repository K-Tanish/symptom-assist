"""
Tests for nlp_extractor.py
"""
import pytest
from app.core.nlp_extractor import SymptomExtractor

@pytest.fixture
def extractor():
    return SymptomExtractor()  # uses manual synonyms fallback

def test_basic_symptom_detection(extractor):
    result = extractor.extract("I have fever and headache")
    assert "fever" in result.symptoms
    assert "headache" in result.symptoms

def test_negation_detection(extractor):
    result = extractor.extract("I have fever but no cough")
    assert "fever" in result.symptoms
    assert "cough" in result.negated
    assert "cough" not in result.symptoms

def test_no_symptoms(extractor):
    result = extractor.extract("I feel fine today")
    assert result.symptoms == []
    assert result.negated == []

def test_multiple_negations(extractor):
    result = extractor.extract("I don't have fever or headache")
    assert "fever" in result.negated
    assert "headache" in result.negated

def test_synonym_matching(extractor):
    result = extractor.extract("I am feeling dizzy and lightheaded")
    assert "dizziness" in result.symptoms

def test_empty_input(extractor):
    result = extractor.extract("")
    assert result.symptoms == []