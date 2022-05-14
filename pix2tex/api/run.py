from multiprocessing import Process
import subprocess
import os
import sys

def start_api(path='.', public=False):
    cmd = ['uvicorn', 'app:app', '--port', '8502']
    if public: cmd += ['--host', '0.0.0.0']
    subprocess.call(cmd, cwd=path)


def start_frontend(path='.'):
    subprocess.call(['streamlit', 'run', 'streamlit.py'], cwd=path)


if __name__ == '__main__':
    path = os.path.realpath(os.path.dirname(__file__))
    public = len(sys.argv) > 1 and sys.argv[1] == 'public'
    api = Process(target=start_api, kwargs={'path': path, 'public': public})
    api.start()
    frontend = Process(target=start_frontend, kwargs={'path': path})
    frontend.start()
    api.join()
    frontend.join()
