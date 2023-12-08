FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

ENV python=3.10

COPY requirements.txt requirements.txt
RUN ["pip", "install","--upgrade","pip"]
RUN ["pip", "install", "-r", "requirements.txt"]

CMD ["jupyter","notebook","--port","6660","--no-browser","--ip","0.0.0.0","--allow-root"]
