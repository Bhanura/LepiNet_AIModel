FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Set permission for the app to write files (important for training)
RUN chmod -R 777 /code

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]