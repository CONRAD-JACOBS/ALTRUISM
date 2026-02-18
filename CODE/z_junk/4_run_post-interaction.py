import random

from questionnaire_core import run_questionnaire

QUESTIONS = [
    {"q": "Based on what you observed today, do you believe that Zork has a mind?", "rtype": "binary", "neg": "No", "pos": "Yes", "max": 0, "neutral_lbl": ""},
    {"q": "How much sympathy do you have for Zork?", "rtype": "scale", "neg": "None", "pos": "A lot", "max": 7, "neutral_lbl": "Jammin"},
    {"q": "Do you wish that your interaction with Zork had been more erotically charged?", "rtype": "binary", "neg": "No", "pos": "Yes", "max": 0, "neutral_lbl": ""},
    {"q": "How much empathy do you have for Zork?", "rtype": "scale", "neg": "None", "pos": "A lot", "max": 9, "neutral_lbl": "Balderdash"},
]

#maybe need more rigorous counterbalancing of questions?
random.shuffle(QUESTIONS)

if __name__ == "__main__":
    run_questionnaire(QUESTIONS, block_id="post_interaction")