import re

def route_question(question):

    q = question.lower().strip()

    # ----------------------------------
    # DOCUMENT QUESTIONS
    # ----------------------------------

    document_keywords = [
        "this pdf",
        "this document",
        "summarize pdf",
        "summarize this pdf",
        "what is this pdf about",
        "what is this document about",
        "from the pdf",
        "from this pdf",
        "from the document",
        "in the pdf",
        "summarize it",
        "explain it",
        "tell me about it",
        "what is it about",
        "skills",
        "experience",
        "education",
        "resume"
    ]

    if any(word in q for word in document_keywords):
        return "RAG"

    # ----------------------------------
    # STATISTICS
    # (checked before MATH because the
    #  math keywords are broad and would
    #  otherwise swallow stats questions)
    # ----------------------------------

    stats_keywords = [
        "mean",
        "median",
        "mode",
        "variance",
        "standard deviation",
        "probability",
        "distribution"
    ]

    if any(word in q for word in stats_keywords):
        return "STATISTICS"

    # ----------------------------------
    # LOGIC
    # (checked before MATH so a "solve this
    #  puzzle" question routes to logic, not
    #  to the math agent)
    # ----------------------------------

    logic_keywords = [
        "puzzle",
        "logic",
        "reasoning",
        "arrangement",
        "circle",
        "seating"
    ]

    if any(word in q for word in logic_keywords):
        return "LOGIC"

    # ----------------------------------
    # MATH
    # ----------------------------------

    math_keywords = [
        "solve",
        "equation",
        "integral",
        "derivative",
        "matrix",
        "algebra",
        "calculate",
        "multiply",
        "divide",
        "addition",
        "subtraction"
    ]

    if any(word in q for word in math_keywords):
        return "MATH"

    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", q):
        return "MATH"

    # ----------------------------------
    # GENERAL KNOWLEDGE
    # ----------------------------------

    general_keywords = [
        "who is",
        "what is",
        "when is",
        "when was",
        "where is",
        "where was",
        "prime minister",
        "president",
        "capital",
        "population",
        "country",
        "india",
        "latest",
        "news",
        "today",
        "current"
    ]

    if any(word in q for word in general_keywords):
        return "GENERAL"

    return "RAG"