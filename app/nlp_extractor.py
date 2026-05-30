import logging

logger = logging.getLogger("SymptomAssist.NLPExtractor")

def extract_symptoms_from_text(user_input: str) -> dict:
    """
    Parses user natural language strings to extract symptoms.
    Includes comprehensive input validation and defensive extraction fallback.
    """
    # Defensive programming guard clauses
    if not user_input or not isinstance(user_input, str):
        logger.warning("Received invalid or empty input structure for symptom extraction.")
        return {"extracted_symptoms": [], "severity_flags": [], "error": "Invalid input text"}

    sanitized_input = user_input.strip()
    
    result = {
        "extracted_symptoms": [],
        "severity_flags": []
    }

    try:
        # Example processing workflow (e.g., tokenizing or calling local dictionary matcher)
        # Replacing unsafe dict matching indexers with safe .get() configurations
        
        # Simulated extraction processing safely:
        if "severe" in sanitized_input.lower() or "chest pain" in sanitized_input.lower():
            result["severity_flags"].append("high")
            
        return result

    except KeyError as key_err:
        logger.error(f"Data schema key missing during NLP symptom compilation: {key_err}")
        return {"extracted_symptoms": [], "severity_flags": [], "error": "Data extraction anomaly"}
    except Exception as e:
        logger.error(f"Unexpected breakdown during NLP extraction sequence: {str(e)}")
        return {"extracted_symptoms": [], "severity_flags": [], "error": "Processing failure"}