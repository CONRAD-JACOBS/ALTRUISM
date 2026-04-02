from datetime import datetime
from local_captcha.captcha_app.captcha_routes import register_captcha_routes
from pathlib import Path
import uuid
from flask import Flask, jsonify, request, make_response, render_template, redirect, send_from_directory
import csv  # if you want header creation here too
import random
import json
import os
import shutil
import sys
import threading
import subprocess
BASE = Path(__file__).resolve().parent
HYPERBASE = Path(__file__).resolve().parent.parent
HYPERHYPERBASE = Path(__file__).resolve().parent.parent.parent
VOICE_CHAT_SESSIONS_DIR = HYPERHYPERBASE / "voice-llm-chat" / "sessions"
ALTRUISM_DATA_DIR = HYPERBASE / "DATA"
VOICE_CHAT_ARTIFACTS = (
    "session_dialogue.txt",
    "conversation_log.jsonl",
    "watchdog_summary.json",
    "session_language_metrics.json",
)

TEST_AUTO_FILL = False
TEST_BYPASS_ROBOT_COMMANDS = False
# In robot stage:
# Press Enter to stop the alert sound. Press Ctrl+Enter to advance manually.


def get_startup_config_warning():
    warnings = []

    if TEST_AUTO_FILL is not False:
        warnings.append("TEST_AUTO_FILL is {} (expected False)".format(TEST_AUTO_FILL))

    if TEST_BYPASS_ROBOT_COMMANDS is not False:
        warnings.append(
            "TEST_BYPASS_ROBOT_COMMANDS is {} (expected False)".format(TEST_BYPASS_ROBOT_COMMANDS)
        )

    pre_config_path = BASE / "local_captcha" / "configs" / "pre_config.json"
    try:
        pre_cfg = json.loads(pre_config_path.read_text(encoding="utf-8"))
        goal_correct = pre_cfg.get("goal_correct")
        if goal_correct != 10:
            warnings.append(
                'pre_config.json "goal_correct" is {} (expected 10)'.format(goal_correct)
            )
    except Exception as exc:
        warnings.append("Could not read pre_config.json: {}".format(exc))

    if not warnings:
        return None

    return (
        "Warning: troubleshooting settings are active for this session.\n\n"
        + "\n".join("- {}".format(item) for item in warnings)
    )

def launch_robot_stack():
    if sys.platform != "darwin":
        print("WARN: Robot stack launcher is macOS-only.")
        return

    script = """tell application "Terminal"
    activate
    delay 0.2

    -- Window 1: uq-neuro-nao Py3 server (create window by running the real command)
    do script "clear; echo '=== UQ-NEURO-NAO Py3 (src_py3.app) ==='; cd {}/uq-neuro-nao && PY3_API_PORT=5001 UQ_PY3_VERBOSE=0 PY3_API_DEBUG=0 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m src_py3.app"
    delay 0.2                                                      

    -- Window 2: bridge_server (prep runs here; no separate PREP window)
    do script "clear; echo '=== VOICE-LLM-CHAT BRIDGE (src.bridge_server) ==='; rm -f {}/voice-llm-chat/sessions/CURRENT_SESSION.txt; echo 'Removed CURRENT_SESSION.txt (if it existed).'; cd {}/voice-llm-chat && BRIDGE_VERBOSE=0 ./.venv/bin/python -m src.bridge_server"
    delay 0.2

    -- Window 3: Py2 worker (wait for bridge port before starting)
    do script "clear; echo '=== UQ-NEURO-NAO Py2 (run_chat_with_bumper) ==='; echo 'Waiting for bridge_server on 127.0.0.1:5055...'; until /usr/bin/nc -z 127.0.0.1 5055; do sleep 0.2; done; echo 'Bridge is up. Launching Py2 worker.'; cd {}/uq-neuro-nao && NAO_WORKER_VERBOSE=0 /Library/Frameworks/Python.framework/Versions/2.7/bin/python -m src_py2.main.run_chat_with_bumper"
end tell
""".format(HYPERHYPERBASE, HYPERHYPERBASE, HYPERHYPERBASE, HYPERHYPERBASE)
    try:
        subprocess.Popen(["osascript", "-e", script])
    except Exception as exc:
        print("WARN: Failed to launch robot stack: {}".format(exc))


def schedule_robot_say_via_py2(text, delay_sec=5.0):
    def _task():
        session_dir = get_latest_voice_chat_session_dir()
        if not session_dir:
            print("WARN: No active voice-chat session found for delayed robot speech.")
            return

        system_inbox_dir = session_dir / "robot_system_inbox"
        system_inbox_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "kind": "say",
            "text": text,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "master_server_delayed_prompt",
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = system_inbox_dir / "system_{}_say.json".format(stamp)
        tmp_path = system_inbox_dir / "system_{}_say.json.tmp".format(stamp)

        try:
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(str(tmp_path), str(out_path))
        except Exception as exc:
            print("WARN: Failed to enqueue delayed robot speech: {}".format(exc))

    timer = threading.Timer(delay_sec, _task)
    timer.daemon = True
    timer.start()
    return timer


def robot_commands_enabled():
    return not TEST_BYPASS_ROBOT_COMMANDS


def get_latest_voice_chat_session_dir():
    current_session_path = VOICE_CHAT_SESSIONS_DIR / "CURRENT_SESSION.txt"
    if current_session_path.exists():
        try:
            raw = current_session_path.read_text(encoding="utf-8").strip()
            if raw:
                session_dir = Path(raw).expanduser()
                if session_dir.is_dir():
                    return session_dir
        except Exception as exc:
            print("WARN: Failed reading CURRENT_SESSION.txt: {}".format(exc))

    session_dirs = [
        p for p in VOICE_CHAT_SESSIONS_DIR.glob("session_*")
        if p.is_dir()
    ]
    if not session_dirs:
        return None
    return max(session_dirs, key=lambda p: p.stat().st_mtime)


def copy_robot_session_artifacts(exp_sid, exp):
    participant_number = int(exp["participant_number"])
    participant_prefix = "P{:03d}".format(participant_number)
    exp_suffix = str(exp_sid)[:8]
    watchdog_total = 0

    ALTRUISM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    source_session_dir = get_latest_voice_chat_session_dir()
    metadata = {
        "exp_sid": exp_sid,
        "participant_number": participant_number,
        "copied_at": datetime.now().isoformat(timespec="milliseconds"),
        "source_session_dir": str(source_session_dir) if source_session_dir else "",
        "copied_files": [],
        "watchdog_total": 0,
        "total_words": None,
        "mean_words_per_turn": None,
        "word_rate_wps": None,
        "robot_total_words": None,
        "robot_mean_words_per_turn": None,
        "robot_word_rate_wps": None,
        "mean_latency_sec": None,
    }

    if source_session_dir is None:
        metadata["status"] = "missing_source_session"
    else:
        metadata["status"] = "ok"
        for filename in VOICE_CHAT_ARTIFACTS:
            src = source_session_dir / filename
            if not src.is_file():
                continue

            dest_name = "{}_{}_{}".format(participant_prefix, exp_suffix, filename)
            dest = ALTRUISM_DATA_DIR / dest_name
            shutil.copy2(src, dest)
            metadata["copied_files"].append(dest.name)

            if filename == "watchdog_summary.json":
                try:
                    summary = json.loads(src.read_text(encoding="utf-8"))
                    watchdog_total = int(summary.get("watchdog_total", 0) or 0)
                except Exception as exc:
                    print("WARN: Failed parsing watchdog summary {}: {}".format(src, exc))
            elif filename == "session_language_metrics.json":
                try:
                    summary = json.loads(src.read_text(encoding="utf-8"))
                    metadata["total_words"] = summary.get("total_words")
                    metadata["mean_words_per_turn"] = summary.get("mean_words_per_turn")
                    metadata["word_rate_wps"] = summary.get("word_rate_wps")
                    metadata["robot_total_words"] = summary.get("robot_total_words")
                    metadata["robot_mean_words_per_turn"] = summary.get("robot_mean_words_per_turn")
                    metadata["robot_word_rate_wps"] = summary.get("robot_word_rate_wps")
                    metadata["mean_latency_sec"] = summary.get("mean_latency_sec")
                except Exception as exc:
                    print("WARN: Failed parsing session language metrics {}: {}".format(src, exc))

    metadata["watchdog_total"] = watchdog_total
    metadata_name = "{}_{}_voice_session_metadata.json".format(participant_prefix, exp_suffix)
    metadata_path = ALTRUISM_DATA_DIR / metadata_name
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    exp["watchdog_total"] = watchdog_total
    exp["voice_session_metadata_path"] = str(metadata_path)
    exp["voice_session_source_dir"] = metadata["source_session_dir"]
    return metadata

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
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_2",
        "type": "likert10",
        "text": "To what extent is the average computer active?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_3",
        "type": "likert10",
        "text": "To what extent do technology devices and machines for manufacturing, entertainment, and productive processes (e.g., cars, computers, television sets) have intentions?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_4",
        "type": "likert10",
        "text": "To what extent does the average fish have free will?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_5",
        "type": "likert10",
        "text": "To what extent is the average cloud good-looking?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_6",
        "type": "likert10",
        "text": "To what extent are pets useful?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_7",
        "type": "likert10",
        "text": "To what extent does the average mountain have free will?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_8",
        "type": "likert10",
        "text": "To what extent is the average amphibian lethargic?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_9",
        "type": "likert10",
        "text": "To what extent does a television set experience emotions?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_10",
        "type": "likert10",
        "text": "To what extent is the average robot good-looking?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_11",
        "type": "likert10",
        "text": "To what extent does the average robot have consciousness?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_12",
        "type": "likert10",
        "text": "To what extent do cows have intentions?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_13",
        "type": "likert10",
        "text": "To what extent does a car have free will?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_14",
        "type": "likert10",
        "text": "To what extent does the ocean have consciousness?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_15",
        "type": "likert10",
        "text": "To what extent is the average camera lethargic?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_16",
        "type": "likert10",
        "text": "To what extent is a river useful?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_17",
        "type": "likert10",
        "text": "To what extent does the average computer have a mind of its own?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_18",
        "type": "likert10",
        "text": "To what extent is a tree active?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_19",
        "type": "likert10",
        "text": "To what extent is the average kitchen appliance useful?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_20",
        "type": "likert10",
        "text": "To what extent does a cheetah experience emotions?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_21",
        "type": "likert10",
        "text": "To what extent does the environment experience emotions?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_22",
        "type": "likert10",
        "text": "To what extent does the average insect have a mind of its own?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_23",
        "type": "likert10",
        "text": "To what extent does a tree have a mind of its own?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_24",
        "type": "likert10",
        "text": "To what extent are technology devices and machines for manufacturing, entertainment, and productive processes (e.g., cars, computers, television sets) durable?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_25",
        "type": "likert10",
        "text": "To what extent is the average cat active?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_26",
        "type": "likert10",
        "text": "To what extent does the wind have intentions?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_27",
        "type": "likert10",
        "text": "To what extent is the forest durable?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_28",
        "type": "likert10",
        "text": "To what extent is a tortoise durable?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_29",
        "type": "likert10",
        "text": "To what extent does the average reptile have consciousness?",
        "anchors": ("Not at all", "Extremely"),
    },
    {
        "id": "idaq_30",
        "type": "likert10",
        "text": "To what extent is the average dog good-looking?",
        "anchors": ("Not at all", "Extremely"),
    },

]

random.shuffle(Q_PRE_IDAQ)

Q_PRE_2050 = [
    {
        "id": "2050_art",
        "type": "likert10",
        "text": "How likely is it that by 2050 some world-class modern art museums will start collecting physical artworks (paintings, sculptures, installations, etc.) both conceived and created by robots?",
        "anchors": ("Not at all", "Extremely"),
    },

    {
        "id": "2050_other_worlds",
        "type": "likert10",
        "text": "How likely is it that by 2050 humanoid robots will visit moons or planets beyond Mars?",
        "anchors": ("Not at all", "Extremely"),
    },

        {
        "id": "2050_rubbish_collectors",
        "type": "likert10",
        "text": "How likely is it that Australian rubbish collectors will be replaced by robots by 2050?",
        "anchors": ("Not at all", "Extremely"),
    },

        {
        "id": "2050_flight_attendants",
        "type": "likert10",
        "text": "How likely is it that flight attendants on Qantas and Virgin Airlines will mostly be replaced by robots by 2050?",
        "anchors": ("Not at all", "Extremely"),
    },

            {
        "id": "2050_experiments",
        "type": "likert10",
        "text": "How likely is that by 2050 most psychology experiments in Australia will be conducted not by human students and lab assistants but by robots?",
        "anchors": ("Not at all", "Extremely"),
    },

                {
        "id": "2050_school",
        "type": "likert10",
        "text": "How likely is it that by 2050 robots will be largely responsible for educating and disciplining children in state-run schools in Australia?",
        "anchors": ("Not at all", "Extremely"),
    },
          {
        "id": "2050_hiking",
        "type": "likert10",
        "text": "How likely is it that in 2050 someone hiking in a popular Australian national park would encounter a robot (drones excluded) on the hiking trail?",
        "anchors": ("Not at all", "Extremely"),
    },
                        {
        "id": "2050_nursing_home",
        "type": "likert10",
        "text": "How likely is it that by 2050 nursing homes in Australia will be staffed more by robots than by humans?",
        "anchors": ("Not at all", "Extremely"),
    },
                            {
        "id": "2050_rights",
        "type": "likert10",
        "text": "How likely is it that by 2050 robots will have certain legal rights in Australia?",
        "anchors": ("Not at all", "Extremely"),
    },
                                {
        "id": "2050_neurosurgery",
        "type": "likert10",
        "text": "How likely is it that by 2050 robots will perform neurosurgery without human invervention?",
        "anchors": ("Not at all", "Extremely"),
    },
                                    {
        "id": "2050_crime",
        "type": "likert10",
        "text": "How likely is it that by 2050 at least one robot will have been convicted of a crime in Australia?",
        "anchors": ("Not at all", "Extremely"),
    },
                                        {
        "id": "2050_sports",
        "type": "likert10",
        "text": "How likely is it that by 2050 humanoid robots will be competitive with humans in team sports, such as football and rugby?",
        "anchors": ("Not at all", "Extremely"),
    },
                                          {
        "id": "2050_mannequins",
        "type": "likert10",
        "text": "How likely is it that by 2050 most mannequins in Australian clothing stores will be intelligent social robots?",
        "anchors": ("Not at all", "Extremely"),
    },
                                                {
        "id": "2050_police",
        "type": "likert10",
        "text": "How likely is it that by 2050 police robots will be responsible for the majority of arrests made in Australia?",
        "anchors": ("Not at all", "Extremely"),
    },
                                                {
        "id": "2050_shelters",
        "type": "likert10",
        "text": "How likely is it that by 2050 there will be shelters in Australia for homeless robots?",
        "anchors": ("Not at all", "Extremely"),
    }
]

random.shuffle(Q_PRE_2050)

Q_POST_GATORS = [
    {
        "id": "gators_1",
        "type": "likert7",
        "subscale": "S1",
        "text": "I can trust persons and organizations related to development of robots.",
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
        "text": "How likeable was Zeke?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
    {
        "id": "robot_empathy",
        "type": "likert7",
        "text": "How much empathy did you feel for Zeke?",
        "anchors": ("None", "Some", "A lot"),
    },
    {
        "id": "robot_friendliness",
        "type": "likert7",
        "text": "How friendly did you find Zeke?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
        {
        "id": "conversation_interestingness",
        "type": "likert7",
        "text": "How interesting did you find your conversation with Zeke?",
        "anchors": ("Not at all", "Somewhat", "Extremely"),
    },
        {
        "id": "mentacy_belief",
        "type": "binary",
        "text": "Based on what you observed today, do you believe Zeke has a mind?",
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
Q_POST_SPECIFIC.insert(mentacy_index + 1, belief_confidence_item)


app = Flask(
    __name__,
    template_folder=str(BASE / "templates"),
    static_folder=str(BASE  / "static"),
)

EXP_SESSIONS = {}  # exp_sid -> dict with participant, csv_path, jsonl_path, etc.

import uuid
from datetime import datetime

def create_new_experiment_session(participant_number, age, gender, results_dir):
    exp_sid = uuid.uuid4().hex
    started = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "P{:03d}_{}_{}.csv".format(participant_number, started, exp_sid[:8])
    jsonl_path = results_dir / "P{:03d}_{}_{}.jsonl".format(participant_number, started, exp_sid[:8])

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
        "watchdog_total": 0,
        "startup_warning": get_startup_config_warning(),
        "startup_warning_shown": False,
    }
    return exp_sid, EXP_SESSIONS[exp_sid]


def get_or_create_experiment_session(participant_number, age, gender, results_dir):
    exp_sid = request.cookies.get("exp_session")
    if exp_sid and exp_sid in EXP_SESSIONS:
        return exp_sid, EXP_SESSIONS[exp_sid]

    exp_sid = uuid.uuid4().hex
    started = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "P{:03d}_{}_{}.jsonl".format(participant_number, started, exp_sid[:8])
    csv_path = results_dir / "P{:03d}_{}_{}.csv".format(participant_number, started, exp_sid[:8])

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
        "watchdog_total": 0,
        "startup_warning": get_startup_config_warning(),
        "startup_warning_shown": False,
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
    distractors_dir=BASE / "local_captcha" / "stimuli" / "DISTRACTORS",
    config_path=BASE / "local_captcha" / "configs" / "pre_config.json",
    start_mode="captcha",
    next_url="/stage/q_pre_captcha",
    EXP_SESSIONS=EXP_SESSIONS,   # <-- ADD THIS
)

register_captcha_routes(
    app,
    stage_id="captcha_post",
    targets_dir=BASE / "local_captcha" / "stimuli" / "TARGETS",
    distractors_dir=BASE / "local_captcha" / "stimuli" / "DISTRACTORS",
    config_path=BASE / "local_captcha" / "configs" / "post_config.json",
    start_mode="captcha",
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
    results_dir=HYPERBASE / "DATA"
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
            schedule_robot_say_via_py2("Oh, hey, I was just dreaming for a moment. Did the experiment start already?", delay_sec=22)

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
    notification_path = BASE / "audio" / "notification.mp3"
    notification_url = None
    if notification_path.exists():
        notification_url = "/audio/notification.mp3"
    else:
        print("WARN: Optional robot-stage notification audio missing: {}".format(notification_path))
    return render_template("robot_stage.html", notification_url=notification_url)

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(BASE / "audio", filename)

@app.route("/api/robot/finish", methods=["POST"])
def robot_finish():
    exp_sid, exp = require_exp_session()
    if not exp_sid:
        return redirect("/")

    if not TEST_BYPASS_ROBOT_COMMANDS:
        try:
            copy_robot_session_artifacts(exp_sid, exp)
        except Exception as exc:
            print("WARN: Failed to copy voice chat artifacts for {}: {}".format(exp_sid, exc))

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
        title="General Post-Questionnaire",
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
        title="Specific Post-Questionnaire",
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
    return (
        "<div style='min-height:100vh;display:flex;align-items:center;justify-content:center;'>"
        "<h2>All done. <br>Your data has been saved. <br> Please inform the experimenter.</h2>"
        "</div>"
    )



if __name__ == "__main__":
    # 127.0.0.1 is always this machine ON ANY MACHINE. It's a standard.
    if robot_commands_enabled() and os.getenv("AUTO_START_ROBOT_STACK", "1") == "1":
        threading.Thread(target=launch_robot_stack, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
