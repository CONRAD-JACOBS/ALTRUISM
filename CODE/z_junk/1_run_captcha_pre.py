from captcha_app.server import create_app
import os
import threading
import webbrowser

def open_browser(url):
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    app = create_app(
        targets_dir="stimuli/TARGETS_pre",
        distractors_dir="stimuli/shared_distractors",
        config_path="configs/pre_config.json",
        results_dir="results/pre",
        auto_open=False
    )
    host = "127.0.0.1"
    port = int(os.environ.get("PORT", "5000"))
    url = "http://{}:{}/".format(host, port)
    threading.Timer(0.5, open_browser, args=(url,)).start()
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
