FROM ashutoshtarun/joint_model:1

WORKDIR /app

RUN pip install flask transformers torch nltk

COPY . .

CMD ["python3", "app.py"]
