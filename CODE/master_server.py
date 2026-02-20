from datetime import datetime
from local_captcha.captcha_app.captcha_routes import register_captcha_routes
from pathlib import Path
import uuid
from flask import Flask, jsonify, request, make_response, render_template, redirect
import csv  # if you want header creation here too
import random
import json
import os
import sys
import threading
import subprocess
BASE = Path(__file__).resolve().parent
HYPERBASE = Path(__file__).resolve().parent.parent
print(HYPERBASE)
HYPERHYPERBASE = Path(__file__).resolve().parent.parent.parent

TEST_AUTO_FILL = False
TEST_BYPASS_ROBOT_COMMANDS = False

def launch_robot_stack():
    if sys.platform != "darwin":
        print("WARN: Robot stack launcher is macOS-only.")
        return

    script = f"""tell application "Terminal"
    activate
    delay 0.2

    -- Window 1: uq-neuro-nao Py3 server (create window by running the real command)
    do script "clear; echo '=== UQ-NEURO-NAO Py3 (src_py3.app) ==='; cd {HYPERHYPERBASE}/uq-neuro-nao && PY3_API_PORT=5001 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m src_py3.app"
    delay 0.2                                                      

    -- Window 2: bridge_server (prep runs here; no separate PREP window)
    do script "clear; echo '=== VOICE-LLM-CHAT BRIDGE (src.bridge_server) ==='; rm -f {HYPERHYPERBASE}/voice-llm-chat/sessions/CURRENT_SESSION.txt; echo 'Removed CURRENT_SESSION.txt (if it existed).'; cd {HYPERHYPERBASE}/voice-llm-chat && ./.venv/bin/python -m src.bridge_server"
    delay 0.2

    -- Window 3: Py2 worker (wait for bridge port before starting)
    do script "clear; echo '=== UQ-NEURO-NAO Py2 (run_chat_with_bumper) ==='; echo 'Waiting for bridge_server on 127.0.0.1:5055...'; until /usr/bin/nc -z 127.0.0.1 5055; do sleep 0.2; done; echo 'Bridge is up. Launching Py2 worker.'; cd {HYPERHYPERBASE}/uq-neuro-nao && /Library/Frameworks/Python.framework/Versions/2.7/bin/python -m src_py2.main.run_chat_with_bumper"
end tell
"""
    try:
        subprocess.Popen(["osascript", "-e", script])
    except Exception as exc:
        print("WARN: Failed to launch robot stack: {}".format(exc))


def schedule_robot_say_via_py2(text, delay_sec=5.0):
    def _task():
        if sys.platform != "darwin":
            print("WARN: Robot say launcher is macOS-only.")
            return
        text_json = json.dumps(text)
        cmd = [
            "/Library/Frameworks/Python.framework/Versions/2.7/bin/python",
            "-c",
            "from src_py2.robot.nao_robot import NAORobot;"
            "from src_py2.robot.conversation_manager import ConversationManager;"
            "r=NAORobot('clas');"
            "c=ConversationManager(r);"
            "txt={};"
            "dur=max(2.5, 0.45*len([w for w in txt.strip().split() if w]));"
            "c.speak_n_gest_next_level([[txt, None, None, dur]], leds=True);".format(text_json),
        ]
        try:
            subprocess.Popen(cmd, cwd=f"{HYPERHYPERBASE}/uq-neuro-nao")
        except Exception as exc:
            print("WARN: Failed to launch py2 say: {}".format(exc))

    timer = threading.Timer(delay_sec, _task)
    timer.daemon = True
    timer.start()
    return timer


def robot_commands_enabled():
    return not TEST_BYPASS_ROBOT_COMMANDS

# STAGES
# captcha_pre
# q_pre_captcha
# q_pre_idaq
# q_pre_2050
# robot
# q_post_gators
# q_post_specific
# captcha_post



# OPTIONS: binary, likert7, likert10
Q_PRE_CAPTCHA = [
    {
        "id": "captcha_task_fun",
        "type": "likert7",
        "text": "How fun did you find the image recognition task?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
        {
        "id": "captcha_task_enjoyment",
        "type": "likert7",
        "text": "Rate your agreement with this statement: 'I enjoyed doing the image recognition task.'",
        "anchors": ("Strongly disagree", "Neutral", "Strongly agree"),
    },
    {
        "id": "general_captcha_liking",
        "type": "likert7",
        "text": "In general, how much do you like doing similar image recognition tasks?",
        "anchors": ("Not at all", "Somewhat", "Very much"),
    },
        {
        "id": "captcha_task_difficulty",
        "type": "likert7",
        "text": "How difficult was the image recognition task?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
            {
        "id": "captcha_task_meaningfulness",
        "type": "likert7",
        "text": "How meaningful was the image recognition task?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
]

random.shuffle(Q_PRE_CAPTCHA)

Q_PRE_IDAQ = [

    {
        "id": "idaq_1",
        "type": "likert10",
        "text": "To what extent is the desert lethargic?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_2",
        "type": "likert10",
        "text": "To what extent is the average computer active?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_3",
        "type": "likert10",
        "text": "To what extent do technology devices and machines for manufacturing, entertainment, and productive processes (e.g., cars, computers, television sets) have intentions?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_4",
        "type": "likert10",
        "text": "To what extent does the average fish have free will?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_5",
        "type": "likert10",
        "text": "To what extent is the average cloud good-looking?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_6",
        "type": "likert10",
        "text": "To what extent are pets useful?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_7",
        "type": "likert10",
        "text": "To what extent does the average mountain have free will?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_8",
        "type": "likert10",
        "text": "To what extent is the average amphibian lethargic?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_9",
        "type": "likert10",
        "text": "To what extent does a television set experience emotions?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_10",
        "type": "likert10",
        "text": "To what extent is the average robot good-looking?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_11",
        "type": "likert10",
        "text": "To what extent does the average robot have consciousness?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_12",
        "type": "likert10",
        "text": "To what extent do cows have intentions?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_13",
        "type": "likert10",
        "text": "To what extent does a car have free will?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_14",
        "type": "likert10",
        "text": "To what extent does the ocean have consciousness?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_15",
        "type": "likert10",
        "text": "To what extent is the average camera lethargic?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_16",
        "type": "likert10",
        "text": "To what extent is a river useful?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_17",
        "type": "likert10",
        "text": "To what extent does the average computer have a mind of its own?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_18",
        "type": "likert10",
        "text": "To what extent is a tree active?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_19",
        "type": "likert10",
        "text": "To what extent is the average kitchen appliance useful?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_20",
        "type": "likert10",
        "text": "To what extent does a cheetah experience emotions?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_21",
        "type": "likert10",
        "text": "To what extent does the environment experience emotions?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_22",
        "type": "likert10",
        "text": "To what extent does the average insect have a mind of its own?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_23",
        "type": "likert10",
        "text": "To what extent does a tree have a mind of its own?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_24",
        "type": "likert10",
        "text": "To what extent are technology devices and machines for manufacturing, entertainment, and productive processes (e.g., cars, computers, television sets) durable?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_25",
        "type": "likert10",
        "text": "To what extent is the average cat active?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_26",
        "type": "likert10",
        "text": "To what extent does the wind have intentions?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_27",
        "type": "likert10",
        "text": "To what extent is the forest durable?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_28",
        "type": "likert10",
        "text": "To what extent is a tortoise durable?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_29",
        "type": "likert10",
        "text": "To what extent does the average reptile have consciousness?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "idaq_30",
        "type": "likert10",
        "text": "To what extent is the average dog good-looking?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },

]

random.shuffle(Q_PRE_IDAQ)

Q_PRE_2050 = [
    {
        "id": "2050_art",
        "type": "likert10",
        "text": "How likely is it that by 2050 some world-class modern art museums will start collecting physical artworks (paintings, sculptures, installations, etc.) both conceived and created by robots?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },

    {
        "id": "2050_other_worlds",
        "type": "likert10",
        "text": "How likely is it that by 2050 humanoid robots will visit moons or planets beyond Mars?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },

        {
        "id": "2050_rubbish_collectors",
        "type": "likert10",
        "text": "How likely is it that Australian rubbish collectors will be replaced by robots by 2050?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },

        {
        "id": "2050_flight_attendants",
        "type": "likert10",
        "text": "How likely is it that flight attendants on Qantas and Virgin Airlines will mostly be replaced by robots by 2050?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },

            {
        "id": "2050_experiments",
        "type": "likert10",
        "text": "How likely is that by 2050 most psychology experiments in Australia will be conducted not by human students and lab assistants but by robots?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },

                {
        "id": "2050_school",
        "type": "likert10",
        "text": "How likely is it that by 2050 robots will be largely responsible for educating and disciplining children in state-run schools in Australia?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
          {
        "id": "2050_hiking",
        "type": "likert10",
        "text": "How likely is it that in 2050 someone hiking in a popular Australian national park would encounter a robot (drones excluded) on the hiking trail?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
                        {
        "id": "2050_nursing_home",
        "type": "likert10",
        "text": "How likely is it that by 2050 nursing homes in Australia will be staffed more by robots than by humans?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
                            {
        "id": "2050_rights",
        "type": "likert10",
        "text": "How likely is it that by 2050 robots will have certain legal rights in Australia?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
                                {
        "id": "2050_neurosurgery",
        "type": "likert10",
        "text": "How likely is it that by 2050 robots will perform neurosurgery without human invervention?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
                                    {
        "id": "2050_crime",
        "type": "likert10",
        "text": "How likely is it that by 2050 at least one robot will have been convicted of a crime in Australia?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
                                        {
        "id": "2050_sports",
        "type": "likert10",
        "text": "How likely is it that by 2050 humanoid robots will be competitive with humans in team sports, such as football and rugby?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
                                          {
        "id": "2050_mannequins",
        "type": "likert10",
        "text": "How likely is it that by 2050 most mannequins in Australian clothing stores will be intelligent social robots?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
                                                {
        "id": "2050_police",
        "type": "likert10",
        "text": "How likely is it that by 2050 police robots will be responsible for the majority of arrests made in Australia?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
                                                {
        "id": "2050_shelters",
        "type": "likert10",
        "text": "How likely is it that by 2050 there will be shelters in Australia for homeless robots?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    }
]

random.shuffle(Q_PRE_2050)

Q_POST_GATORS = [
    {
        "id": "gators_1",
        "type": "likert7",
        "subscale": "S1",
        "text": "I can trust persons and organizations related to development of robot.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
        {
        "id": "gators_2",
        "type": "likert7",
        "subscale": "S1",
        "text": "Persons and organizations related to development of robots will consider the needs, thoughts and feelings of their users.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
            {
        "id": "gators_3",
        "type": "likert7",
        "subscale": "S1",
        "text": "I can trust a robot.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
            {
        "id": "gators_4",
        "type": "likert7",
        "subscale": "S1",
        "text": "I would feel relaxed talking with a robot.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
            {
        "id": "gators_5",
        "type": "likert7",
        "subscale": "S1",
        "text": "If robots had emotions, I would be able to befriend them.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                {
        "id": "gators_6",
        "type": "likert7",
        "subscale": "S2",
        "text": "I would feel uneasy if I was given a job where I had to use robots.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                    {
        "id": "gators_7",
        "type": "likert7",
        "subscale": "S2",
        "text": "I fear that a robot would not understand my commands.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                    {
        "id": "gators_8",
        "type": "likert7",
        "subscale": "S2",
        "text": "Robots scare me.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                    {
        "id": "gators_9",
        "type": "likert7",
        "subscale": "S2",
        "text": "I would feel very nervous just being around a robot.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                    {
        "id": "gators_10",
        "type": "likert7",
        "subscale": "S2",
        "text": "I don’t want a robot to touch me.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                        {
        "id": "gators_11",
        "type": "likert7",
        "subscale": "S3",
        "text": "Robots are necessary because they can do jobs that are too hard or too dangerous for people.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_12",
        "type": "likert7",
        "subscale": "S3",
        "text": "Robots can make life easier.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_13",
        "type": "likert7",
        "subscale": "S3",
        "text": "Assigning routine tasks to robots lets people do more meaningful tasks.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_14",
        "type": "likert7",
        "subscale": "S3",
        "text": "Dangerous tasks should primarily be given to robots.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_15",
        "type": "likert7",
        "subscale": "S3",
        "text": "Robots are a good thing for society, because they help people.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_16",
        "type": "likert7",
        "subscale": "S4",
        "text": "Robots may make us even lazier.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_17",
        "type": "likert7",
        "subscale": "S4",
        "text": "Widespread use of robots is going to take away jobs from people.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_18",
        "type": "likert7",
        "subscale": "S4",
        "text": "I am afraid that robots will encourage less interaction between humans.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_19",
        "type": "likert7",
        "subscale": "S4",
        "text": "Robotics is one of the areas of technology that needs to be closely monitored.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
                            {
        "id": "gators_20",
        "type": "likert7",
        "subscale": "S4",
        "text": "Unregulated use of robotics can lead to societal upheavals.",
        "anchors": ("Strongly Disagree", "Neutral", "Strongly Agree"),
    },
     
     
]

random.shuffle(Q_POST_GATORS)

Q_POST_SPECIFIC = [
    {
        "id": "robot_likeability",
        "type": "likert7",
        "text": "How likeable was Zeek?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "robot_empathy",
        "type": "likert7",
        "text": "How much empathy did you feel for Zeek?",
        "anchors": ("None", "Some", "A lot"),
    },
    {
        "id": "robot_friendliness",
        "type": "likert7",
        "text": "How friendly did you find the Zeek?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
        {
        "id": "conversation_interestingness",
        "type": "likert7",
        "text": "How interesting did you find your conversation with Zeek?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
        {
        "id": "mentacy_belief",
        "type": "binary",
        "text": "Based on what you observed today, do you believe Zeek has a mind?",
        "labels": ("No", "Yes"),
    }     
]

# Ensure that the confidence rating always immediately follow the belief question.
belief_confidence_item =   {
        "id": "belief_confidence",
        "type": "likert7",
        "text": "How confident are you in the belief you indicated in the previous question?",
        "anchors": ("Not confident", "Somewhat", "Very confident"),
    }
random.shuffle(Q_POST_SPECIFIC)
mentacy_index = ''
for index, item in enumerate(Q_POST_SPECIFIC):    
    if item['id'] == "mentacy_belief":
        mentacy_index = index
print(mentacy_index)
Q_POST_SPECIFIC.insert(mentacy_index + 1, belief_confidence_item)


app = Flask(
    __name__,
    template_folder=str(BASE / "templates"),
    static_folder=str(BASE  / "static"),
)

EXP_SESSIONS = {}  # exp_sid -> dict with participant, csv_path, jsonl_path, etc.

import uuid
from datetime import datetime

def create_new_experiment_session(*, participant_number: int, age: int, gender: str, results_dir):
    exp_sid = uuid.uuid4().hex
    started = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"P{participant_number:03d}_{started}_{exp_sid[:8]}.csv"
    jsonl_path = results_dir / f"P{participant_number:03d}_{started}_{exp_sid[:8]}.jsonl"

    if not csv_path.exists():
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "exp_sid",
                    "stage_id",
                    "participant_number",
                    "age",
                    "gender",
                    "timestamp_display",
                    "timestamp_submit",
                    "rt_sec",
                    "captcha_index",
                    "correct",
                    "prompt_text",
                    "shown_filenames_json",
                    "shown_labels_json",
                    "selected_indices_json",
                    "selected_filenames_json",
                    "targets_remaining_after",
                    "total_seen_so_far",
                    "total_correct_so_far",
                    "goal_correct",
                    "event",
                    "questionnaire_json"
                ])


    EXP_SESSIONS[exp_sid] = {
        "participant_number": participant_number,
        "age": age,
        "gender": gender,
        "created_at": started,
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }
    return exp_sid, EXP_SESSIONS[exp_sid]


def get_or_create_experiment_session(participant_number: int, age: int, gender: str, results_dir: Path):
    exp_sid = request.cookies.get("exp_session")
    if exp_sid and exp_sid in EXP_SESSIONS:
        return exp_sid, EXP_SESSIONS[exp_sid]

    exp_sid = uuid.uuid4().hex
    started = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / f"P{participant_number:03d}_{started}_{exp_sid[:8]}.jsonl"
    csv_path = results_dir / f"P{participant_number:03d}_{started}_{exp_sid[:8]}.csv"

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "exp_sid",
                "stage_id",
                "participant_number",
                "age",
                "gender",
                "timestamp_display",
                "timestamp_submit",
                "rt_sec",
                "captcha_index",
                "correct",
                "prompt_text",
                "shown_filenames_json",
                "shown_labels_json",
                "selected_indices_json",
                "selected_filenames_json",
                "targets_remaining_after",
                "total_seen_so_far",
                "total_correct_so_far",
                "goal_correct",
                "event",
                "questionnaire_json"
            ])

    EXP_SESSIONS[exp_sid]= {
        "participant_number": participant_number,
        "age": age,
        "gender": gender,
        "created_at": started,
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }
    return exp_sid, EXP_SESSIONS[exp_sid]

def require_exp_session():
    exp_sid = request.cookies.get("exp_session")
    if not exp_sid or exp_sid not in EXP_SESSIONS:
        return None, None
    return exp_sid, EXP_SESSIONS[exp_sid]

# --- Register stages (PRE then POST) ---
register_captcha_routes(
    app,
    stage_id="captcha_pre",
    targets_dir=BASE / "local_captcha" / "stimuli" / "TARGETS",
    distractors_dir=BASE / "local_captcha" / "stimuli" / "shared_distractors",
    config_path=BASE / "local_captcha" / "configs" / "pre_config.json",
    start_mode="chooser",
    next_url="/stage/q_pre_captcha",
    EXP_SESSIONS=EXP_SESSIONS,   # <-- ADD THIS
)

register_captcha_routes(
    app,
    stage_id="captcha_post",
    targets_dir=BASE / "local_captcha" / "stimuli" / "TARGETS",
    distractors_dir=BASE / "local_captcha" / "stimuli" / "shared_distractors",
    config_path=BASE / "local_captcha" / "configs" / "post_config.json",
    start_mode="chooser",
    next_url="/done",
    EXP_SESSIONS=EXP_SESSIONS,   # <-- ADD THIS
)

# --- Global routes ---

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("start.html", error=None)

    raw = (request.form.get("participant_number") or "").strip()
    if raw == "":
        return render_template("start.html", error="Please enter a participant number.")

    try:
        participant_number = int(raw)
        if participant_number <= 0:
            raise ValueError()
    except ValueError:
        return render_template("start.html", error="Participant number must be a positive whole number.")
    
    raw = (request.form.get("age") or "").strip()
    try:
        age = int(raw)
        if age <= 0:
            raise ValueError()
    except ValueError:
        return render_template("start.html", error="Age must be a positive whole number.")

    gender = request.form["gender"]

    exp_sid, exp = create_new_experiment_session(
    participant_number=participant_number,
    age=age,
    gender=gender,
    results_dir=HYPERBASE / "DATA/2_lab"
)


    resp = make_response(redirect("/stage/captcha_pre"))
    resp.set_cookie("exp_session", exp_sid, samesite="Lax")
    return resp

@app.route("/stage/q_pre_captcha", methods=["GET"])
def q_pre_captcha_page():
    exp_sid, exp = require_exp_session()
    if not exp_sid:
        return redirect("/")

    # record display time ONCE
    if "q_pre_captcha_display_ts" not in exp:
        exp["q_pre_captcha_display_ts"] = datetime.now().isoformat(timespec="milliseconds")

    return render_template(
        "questionnaire.html",
        title="Image Recognition Questionnaire",
        questions=Q_PRE_CAPTCHA,
        submit_url="/api/q_pre_captcha/submit",
        error=None,
        auto_fill=TEST_AUTO_FILL,
    )

@app.route("/api/q_pre_captcha/submit", methods=["POST"])
def q_pre_captcha_submit():
    exp_sid, exp = require_exp_session()

    payload = request.get_json(force=True) or {}
    responses = payload.get("responses")

    if not isinstance(responses, dict):
        abort(400, "Invalid questionnaire payload")

    timestamp_submit = datetime.now().isoformat(timespec="milliseconds")
    timestamp_display = exp.get("q_pre_captcha_display_ts")

    if not timestamp_display:
        abort(500, "Missing q_pre_captcha display timestamp")

    # compute RT in seconds
    t0 = datetime.fromisoformat(timestamp_display)
    t1 = datetime.fromisoformat(timestamp_submit)
    rt_sec = (t1 - t0).total_seconds()

    with open(exp["csv_path"], "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            exp_sid,
            "q_pre_captcha",
            exp["participant_number"],
            exp["age"],
            exp["gender"],
            timestamp_display,
            timestamp_submit,
            rt_sec,
            "", "", "", "", "", "", "", "", "", "","",
            "submit",
            json.dumps(responses, ensure_ascii=False)
        ])

    return jsonify({"ok": True, "next_url": "/stage/q_pre_idaq"})


@app.route("/stage/q_pre_idaq", methods=["GET"])
def q_pre_idaq_page():
    exp_sid, exp = require_exp_session()
    if not exp_sid:
        return redirect("/")

    # record display time ONCE
    if "q_pre_idaq_display_ts" not in exp:
        exp["q_pre_idaq_display_ts"] = datetime.now().isoformat(timespec="milliseconds")
    return render_template(
        "questionnaire.html",
        title="General Pre-Questionnaire",
        questions=Q_PRE_IDAQ,
        submit_url="/api/q_pre_idaq/submit",
        error=None,
        auto_fill=TEST_AUTO_FILL,
    )

@app.route("/api/q_pre_idaq/submit", methods=["POST"])
def q_pre_idaq_submit():
    exp_sid, exp = require_exp_session()

    payload = request.get_json(force=True) or {}
    responses = payload.get("responses")

    if not isinstance(responses, dict):
        abort(400, "Invalid questionnaire payload")

    timestamp_submit = datetime.now().isoformat(timespec="milliseconds")
    timestamp_display = exp.get("q_pre_idaq_display_ts")

    if not timestamp_display:
        abort(500, "Missing q_pre_idaq display timestamp")

    # compute RT in seconds
    t0 = datetime.fromisoformat(timestamp_display)
    t1 = datetime.fromisoformat(timestamp_submit)
    rt_sec = (t1 - t0).total_seconds()

    with open(exp["csv_path"], "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            exp_sid,
            "q_pre_idaq",
            exp["participant_number"],
            exp["age"],
            exp["gender"],
            timestamp_display,
            timestamp_submit,
            rt_sec,
            "", "", "", "", "", "", "", "", "", "","",
            "submit",
            json.dumps(responses, ensure_ascii=False)
        ])

    return jsonify({"ok": True, "next_url": "/stage/q_pre_2050"})


@app.route("/stage/q_pre_2050", methods=["GET"])
def q_pre_2050_page():
    exp_sid, exp = require_exp_session()
    if not exp_sid:
        return redirect("/")

    # record display time ONCE
    if "q_pre_2050_display_ts" not in exp:
        exp["q_pre_2050_display_ts"] = datetime.now().isoformat(timespec="milliseconds")

    if "q_pre_captcha_robot_test" not in exp:
        exp["q_pre_captcha_robot_test"] = True
        if robot_commands_enabled():
            schedule_robot_say_via_py2("Oh, hey, did the experiment start already?", delay_sec=12.5)

    return render_template(
        "questionnaire.html",
        title="2050 Questionnaire",
        questions=Q_PRE_2050,
        submit_url="/api/q_pre_2050/submit",
        error=None,
        auto_fill=TEST_AUTO_FILL,
    )

@app.route("/api/q_pre_2050/submit", methods=["POST"])
def q_pre_2050_submit():
    exp_sid, exp = require_exp_session()

    payload = request.get_json(force=True) or {}
    responses = payload.get("responses")

    if not isinstance(responses, dict):
        abort(400, "Invalid questionnaire payload")

    timestamp_submit = datetime.now().isoformat(timespec="milliseconds")
    timestamp_display = exp.get("q_pre_2050_display_ts")

    if not timestamp_display:
        abort(500, "Missing q_pre_2050 display timestamp")

    # compute RT in seconds
    t0 = datetime.fromisoformat(timestamp_display)
    t1 = datetime.fromisoformat(timestamp_submit)
    rt_sec = (t1 - t0).total_seconds()

    with open(exp["csv_path"], "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            exp_sid,
            "q_pre_2050",
            exp["participant_number"],
            exp["age"],
            exp["gender"],
            timestamp_display,
            timestamp_submit,
            rt_sec,
            "", "", "", "", "", "", "", "", "", "","",
            "submit",
            json.dumps(responses, ensure_ascii=False)
        ])

    next_url = "/stage/q_post_gators" if TEST_BYPASS_ROBOT_COMMANDS else "/stage/robot"
    return jsonify({"ok": True, "next_url": next_url})



@app.route("/stage/robot")
def robot_stage():
    if TEST_BYPASS_ROBOT_COMMANDS:
        return redirect("/stage/q_post_gators")
    return render_template("robot_stage.html")

@app.route("/api/robot/finish", methods=["POST"])
def robot_finish():
    # Later: integrate your robot conversation shutdown here
    # (and write stage-end timestamps to your participant csv/jsonl if desired)
    return jsonify({"ok": True, "next_url": "/stage/q_post_gators"})

@app.route("/stage/q_post_gators", methods=["GET"])
def q_post_gators_page():
    exp_sid, exp = require_exp_session()
    if not exp_sid:
        return redirect("/")

    # record display time ONCE
    if "q_post_gators_display_ts" not in exp:
        exp["q_post_gators_display_ts"] = datetime.now().isoformat(timespec="milliseconds")

    saved = exp.get("questionnaires", {}).get("q_post_gators", {})
    return render_template(
        "questionnaire.html",
        title="General Questionnaire",
        questions=Q_POST_GATORS,
        submit_url="/api/q_post_gators/submit",
        error=None,
        responses=saved,
        auto_fill=TEST_AUTO_FILL,
    )


@app.route("/api/q_post_gators/submit", methods=["POST"])
def q_post_gators_submit():
    exp_sid, exp = require_exp_session()
    if not exp_sid:
        return redirect("/")

    payload = request.get_json(force=True) or {}
    responses = payload.get("responses")

    if not isinstance(responses, dict):
        abort(400, "Invalid questionnaire payload")

    timestamp_submit = datetime.now().isoformat(timespec="milliseconds")
    timestamp_display = exp.get("q_post_gators_display_ts")

    if not timestamp_display:
        abort(500, "Missing q_post_gators display timestamp")

    # compute RT in seconds
    t0 = datetime.fromisoformat(timestamp_display)
    t1 = datetime.fromisoformat(timestamp_submit)
    rt_sec = (t1 - t0).total_seconds()

    with open(exp["csv_path"], "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            exp_sid,
            "q_post_gators",
            exp["participant_number"],
            exp["age"],
            exp["gender"],
            timestamp_display,
            timestamp_submit,
            rt_sec,
            "", "", "", "", "", "", "", "", "", "","",
            "submit",
            json.dumps(responses, ensure_ascii=False)
        ])

    return jsonify({"ok": True, "next_url": "/stage/q_post"})

@app.route("/stage/q_post", methods=["GET"])
def q_post_page():
    exp_sid, exp = require_exp_session()
    if not exp_sid:
        return redirect("/")

    # record display time ONCE
    if "q_post_specific_display_ts" not in exp:
        exp["q_post_specific_display_ts"] = datetime.now().isoformat(timespec="milliseconds")

    return render_template(
        "questionnaire.html",
        title="Specific Questionnaire",
        questions=Q_POST_SPECIFIC,
        submit_url="/api/q_post/submit",
        error=None,
        auto_fill=TEST_AUTO_FILL,
    )

@app.route("/api/q_post/submit", methods=["POST"])
def q_post_submit():
    exp_sid, exp = require_exp_session()
    if not exp_sid:
        return redirect("/")

    payload = request.get_json(force=True) or {}
    responses = payload.get("responses")

    if not isinstance(responses, dict):
        abort(400, "Invalid questionnaire payload")

    timestamp_submit = datetime.now().isoformat(timespec="milliseconds")
    timestamp_display = exp.get("q_post_specific_display_ts")

    if not timestamp_display:
        abort(500, "Missing q_post_specific display timestamp")

    # compute RT in seconds
    t0 = datetime.fromisoformat(timestamp_display)
    t1 = datetime.fromisoformat(timestamp_submit)
    rt_sec = (t1 - t0).total_seconds()

    with open(exp["csv_path"], "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            exp_sid,
            "q_post_specific",
            exp["participant_number"],
            exp["age"],
            exp["gender"],
            timestamp_display,
            timestamp_submit,
            rt_sec,
            "", "", "", "", "", "", "", "", "", "", "",
            "submit",
            json.dumps(responses, ensure_ascii=False)
        ])

    return jsonify({"ok": True, "next_url": "/stage/captcha_post"})


@app.route("/done")
def done():
    return "<h2>All done. Thank you.</h2>"



if __name__ == "__main__":
    # 127.0.0.1 is always this machine ON ANY MACHINE. It's a standard.
    if robot_commands_enabled() and os.getenv("AUTO_START_ROBOT_STACK", "1") == "1":
        threading.Thread(target=launch_robot_stack, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
