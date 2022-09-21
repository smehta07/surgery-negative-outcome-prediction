# load python as the parent image
FROM tiangolo/uwsgi-nginx-flask:python3.6

# set up the working directory for the app
WORKDIR /app

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Copy the rest of the app
COPY ./app /app

# install the requirements
RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

# run the app
CMD [ "app.py" ] && flask run -h 0.0.0.0 -p 5000

#CMD [ "app.py" ] && flask db upgrade && flask run -h 0.0.0.0 -p 5000