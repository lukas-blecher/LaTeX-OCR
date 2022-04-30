from multiprocessing import Process
import subprocess
import os


def start_api(path='.'):
    subprocess.call(['uvicorn', 'app:app', '--port', '8502'], cwd=path)


def start_frontend(path='.'):
    subprocess.call(['streamlit', 'run', 'streamlit.py'], cwd=path)


if __name__ == '__main__':
    path = os.path.realpath(os.path.dirname(__file__))
    api = Process(target=start_api, kwargs={'path': path})
    api.start()
    frontend = Process(target=start_frontend, kwargs={'path': path})
    frontend.start()
    api.join()
    frontend.join()
