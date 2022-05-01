FROM python:3.8-slim
RUN pip install torch>=1.7.1
RUN pip install pix2tex[api]

ENTRYPOINT ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8502"]
