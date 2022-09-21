# MLOrthoProj
A repository for the MSc Summer project - ML with Orthopaedic Data

The app is held in a Docker container, the Dockerfile and requirements.txt are for Docker. 

Dockerfile - for the docker environment

requirements.txt - the modules which are needed for the app

venv - the virtual environment



The app folder then contains the code for the application.

app.py - the main python file 

db.env - contains the connection string to the SQL database

__init__.py - declares folder as a package

files - contains the data csv files and models

templates - contains HTML pages

static - static files to do with styling

migrations - if the structure of the SQL datbase is changed in the code, this folder migrates the changes to the database
