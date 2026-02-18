from captcha_app.server import create_app
import webbrowser
import threading

def open_browser(url):
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    app = create_app(
        targets_dir="stimuli/TARGETS_post",
        distractors_dir="stimuli/shared_distractors",
        config_path="configs/post_config.json",
        results_dir="results/post",
        auto_open=False
    )
    host, port = "127.0.0.1", 5000
    url = "http://{}:{}/".format(host, port)
    threading.Timer(0.5, open_browser, args=(url,)).start()
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
