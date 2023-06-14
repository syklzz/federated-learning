FROM intel/openfl
RUN pip install --no-cache-dir --upgrade pip && pip install Pillow &&\
    pip install tensorflow==2.8.0
RUN mkdir /app
COPY . /app
WORKDIR /app
ENTRYPOINT ["fx", "collaborator", "start", "-n", "2"]
