from local_captcha.captcha_app.server import create_app
from pathlib import Path
import threading
import webbrowser

BASE = Path(__file__).resolve().parent

STAGES = [
  {"id": "captcha_pre",  "type": "captcha", "pause_before": True,  "next": "captcha_post",
   "targets": "stimuli/TARGETS_pre", "config": "local_captcha/configs/pre_config.json"},
  {"id": "captcha_post", "type": "captcha", "pause_before": True,  "next": "done",
   "targets": "stimuli/TARGETS_post", "config": "local_captcha/configs/post_config.json"},
]


class Run_class:
    def __init__(self):
        self.targets_dir_prefix = "local_captcha/stimuli/TARGETS_"
        self.distractors_dir = "local_captcha/stimuli/shared_distractors"
        self.config_path_prefix = "local_captcha/configs"   # <- no trailing slash
        self.results_dir = "local_captcha/results"

        self.host = "127.0.0.1"
        self.port = 5000
        self.url = f"http://{self.host}:{self.port}/"

        self.participant_number = 666

    def open_browser(self, url):
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def _1_captcha_pre(self):

        
        app = create_app(
            targets_dir=str(BASE / (self.targets_dir_prefix + "pre")),
            distractors_dir=str(BASE / self.distractors_dir),
            config_path=str(BASE / self.config_path_prefix / "pre_config.json"),
            results_dir=str(BASE / self.results_dir),
            participant_number=self.participant_number,
            auto_open=False,
        )
        # Optional while debugging:
        threading.Timer(0.5, self.open_browser, args=(self.url,)).start()
        app.run(host=self.host, port=self.port, debug=False)

    def _4_captcha_post(self):
        app = create_app(
            targets_dir=str(BASE / (self.targets_dir_prefix + "post")),
            distractors_dir=str(BASE / self.distractors_dir),
            config_path=str(BASE / self.config_path_prefix / "post_config.json"),
            results_dir=str(BASE / self.results_dir),
            participant_number=self.participant_number,
            auto_open=False,
        )
        # threading.Timer(0.5, self.open_browser, args=(self.url,)).start()
        app.run(host=self.host, port=self.port, debug=False)

session = Run_class()
session.participant_number = int(input("Input the participant number:\n"))
print(f"You input this number: {session.participant_number}. \nRestart if incorrect.")

start_with = input("""
Choose a stage number or press ENTER.\n
1) captcha_pre;\n2) captcha_pre_Qs; \n3) robochat; \n4) captcha_post; \n5) captcha_post_Qs\n\n""")

if start_with == "" or start_with == "1":
    session._1_captcha_pre()
    session._4_captcha_post()
elif start_with == "4":
    session._4_captcha_post()
else:
    print("Invalid choice. Start Over")
