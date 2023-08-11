#get python
FROM python:3.9-slim-buster

# create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /opt/spamW2V/

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/spamW2V"

COPY docker-requirements.txt .
RUN pip3 install -r docker-requirements.txt

COPY . .

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]