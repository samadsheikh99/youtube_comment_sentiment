FROM python:3.9-alpine 
RUN apk add --no-cache gcc musl-dev libffi-dev openssl-dev g++
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py