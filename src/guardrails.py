BLOCKED_WORDS = [
    "hack",
    "bomb",
    "malware",
    "exploit"
]

def safety_check(question):

    q = question.lower()

    for word in BLOCKED_WORDS:
        if word in q:
            return False

    return True