FROM tensorflow/tensorflow

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip install tensorflow --upgrade --force-reinstall

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

EXPOSE 8501

WORKDIR /app
COPY . .

CMD ["streamlit", "run", "streamlit_app.py"]