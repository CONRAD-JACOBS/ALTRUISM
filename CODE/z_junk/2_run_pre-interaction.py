import random

from questionnaire_core import run_questionnaire

QUESTIONS = [
    {"q": "How much did you enjoy the reCAPTCHA task?", "rtype": "scale", "neg": "Not at all", "pos": "A lot", "max": 7, "neutral_lbl": "Some"},
    {"q": "Rate how much you agree with this statement: 'I like solving reCAPTCHAs.'", "rtype": "scale", "neg": "Strongly Disagree", "pos": "Strongly Agree", "max": 7, "neutral_lbl": "Somewhat Agree"},
]

#maybe need more rigorous counterbalancing of questions?
random.shuffle(QUESTIONS)

if __name__ == "__main__":
    run_questionnaire(QUESTIONS, block_id="pre_interaction")
