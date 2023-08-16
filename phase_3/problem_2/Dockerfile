FROM python:3.9-slim

RUN apt-get update

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

COPY . /problem_2
WORKDIR /problem_2

EXPOSE 5040

CMD ["python", "src/model_predictor.py"]