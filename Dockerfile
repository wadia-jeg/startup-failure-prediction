FROM python:3.13-slim

RUN pip install pipenv

# Creates a working diretory and cds into it
WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

# Installing dependencies direclty into the system insteead of a virtual environment 
RUN pipenv install --system --deploy

COPY ["predict.py", "models.bin", "./"]

EXPOSE 8088

ENTRYPOINT ["python", "predict.py"]

# 1. Build image 
# docker build -t <IMAGE-NAME> .

#2. Run container
# docker run -it -p 8088:8088 --name <CONTAINER-NAME> <IMAGE-NAME>