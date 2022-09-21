# ====================================================================================================================
# This file contains the routes and methods for the app
# ====================================================================================================================

# ==================================================================
# Imports
# ==================================================================

import os, uuid, sys, warnings, inspect
from flask import Flask, request, redirect, url_for, make_response, render_template, send_from_directory, flash, session
from werkzeug.utils import secure_filename
import string, random, requests
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
import datetime, time, linecache
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from flask_sqlalchemy import SQLAlchemy, sqlalchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash

# ==================================================================
# Config Area 
# Initialises the app, upload folder and database
# ==================================================================

# initialise Flask app
app = Flask(__name__, instance_relative_config=True)

UPLOAD_FOLDER = '/app/files'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

# configure the upload folder and the database used
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SECRET_KEY'] = 'dhjfbhabjknS'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://%s:%s@%s/%s' % (
    os.environ['DBUSER'], os.environ['DBPASS'], os.environ['DBHOST'], os.environ['DBNAME']
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# initialise db connection
db = SQLAlchemy(app)
# initialise database migration management
migrate = Migrate(app, db)

# create a model for the table users 
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    pwdhash = db.Column(db.String(128))
    usertype = db.Column(db.String(5))
    securityqn = db.Column(db.String(100))
    securityans = db.Column(db.String(64))
  
    def __repr__(self):
        return '<User {}>'.format(self.username) 

# ==================================================================
# Helper Functions
# ==================================================================

# Defines the allowed file types for upload
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# General function for training a model and generating accuracy score
# This code is adapted from the following documentation example:
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#
# sphx-glr-auto-examples-text-document-classification-20newsgroups-py
def algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2):
    # fit the data to the model
    clf.fit(X_train, y_train)
    
    # need to persist the model for later
    modelname = name
    full_path_to_model = '/app/files/'+ name + '.pkl'
    joblib.dump(clf, full_path_to_model)
    
    # make predictions using the test input data
    pred = clf.predict(X_test)
    
    # compute some metrics to assess the accuracy of the predictions using this algorithm
    roc = roc_auc_score(y_test, pred)

    # prediction using non-upsampled data to compute metrics for this
    pred2 = clf.predict(X_test2)
    class_report = metrics.precision_recall_fscore_support(y_test2, pred2, average=None)
    confusionmtx = metrics.confusion_matrix(y_test2, pred2)
    
    # return the metrics 
    return roc, class_report, confusionmtx

# Method to process the inputted values on the predict pages to numerical values
def process_one_line(line):
  line = [l.replace('NTGH', '1').replace('HEXH', '2').replace('WANS', '3').replace('NSEC', '4').replace('W', '0').replace('HIP', '1').replace('KNEE', '2').replace('T', '1').replace('F', '0').replace('M', '1').replace('Y', '1').replace('N', '0') for l in line]
  return line

# Method within adminpredictcompafter and userpredictcomplications to retrive the values from the file
def get_rates_values(line):
  split_line = line.split("_")
  
  tp = split_line[3]
  fp = split_line[4]
  tn = split_line[5]
  fn = split_line[6]
  
  roc = split_line[7]

  precision1 = split_line[8]
  precision2 = split_line[9]
  
  recall1 = split_line[10]
  recall2 = split_line[11]

  value0 = split_line[12]
  value1 = split_line[13].strip()
  value0 = float(value0)
  value1 = float(value1)

  return tp, fp, tn, fn, roc, precision1, precision2, recall1, recall2, value0, value1

# Method within adminpredictcompafter and userpredictcomplications to make a prediction
def predict_model(model, values, value0, value1, line):
  model_prediction = model.predict(values)

  # make predictions on the up-sampled data
  prob_prediction = model.predict_proba(values)
  class1_prediction = prob_prediction[:,1]
  class0_prediction = prob_prediction[:,0]
  
  if (str(class1_prediction) == '[0.]'):
    final_pred_class1 = [0]
    final_pred_class0 = [1]
  
  else: 
    # Get mappings
    # Concept from http://blog.data-miners.com/2009/09/adjusting-for-oversampling.html
    class0, class1 = calc_prob(value0, value1)

    # map the prediction made on the up-sampled data to the original
    class1_prediction_mapped = class1_prediction / class1
    class0_prediction_mapped = class0_prediction / class0

    final_pred_class1 = class1_prediction_mapped / (class1_prediction_mapped + class0_prediction_mapped)
    final_pred_class1 = np.round(final_pred_class1, 5)
    final_pred_class1 = final_pred_class1*100
    final_pred_class0 = class0_prediction_mapped / (class1_prediction_mapped + class0_prediction_mapped)
    final_pred_class0 = np.round(final_pred_class0, 5)

  
  split_line = line.split("_")

  final_pred = "This patient has a "
  final_pred += str(final_pred_class1)[1:-1]
  final_pred += "%"
  final_pred += " chance of having this outcome. Since the average risk for this complication is "
  final_pred += str(split_line[14])
  final_pred += "%"
  final_pred += " any risk above "
  final_pred += str(split_line[15].strip())
  final_pred += "%"
  final_pred += " is considered to be HIGH risk compared to the average population."
  
  return final_pred

# def to calculate the non-upsampled probability 
def calc_prob(value0, value1):
    
    # calculate total number of rows
    total = value0 + value1
    
    # calculate the proportions of each class
    prop_0 = value0 / total
    prop_1 = value1 / total
    
    # calculate a mapping from the upsampled data to original
    class0 = 0.5 / prop_0
    class1 = 0.5 / prop_1
    
    return class0, class1

# ==================================================================
# Login - user logs in and is directed to admin/user homepage
# This code is adapted from the following reference:
# https://techmonger.github.io/10/flask-simple-authentication/
# ==================================================================

@app.route('/', methods=['GET', 'POST'])
def login():
  if request.method == "POST":
    username = request.form['username']
    password = request.form['password']

    if not (username and password):
        flash("Username or Password cannot be empty.")
        return redirect(url_for('login'))
    else:
        username = username.strip()
        password = password.strip()

    user = User.query.filter_by(username=username).first()

    try:
      if user and check_password_hash(user.pwdhash, password):
          session['username'] = username
          if user.usertype == 'admin':
            return redirect(url_for("homepage"))
          else:
            return redirect(url_for("userhomepage"))
      else:
        flash("Invalid username or password. If you haven't Signed up click 'Sign up!'")
    except:
      flash("Invalid username or password. If you haven't Signed up click 'Sign up!'")
      return redirect(url_for('login'))

  return render_template("login.html")

# ==================================================================
# Logout - user logs out
# This code is adapted from the following reference:
# https://www.tutorialspoint.com/flask/flask_sessions.htm
# ==================================================================
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Successfully logged out")
    return redirect(url_for('login'))

# ==================================================================
# Register - user registers as either admin/user
# This code is adapted from the following reference:
# https://techmonger.github.io/10/flask-simple-authentication/
# ==================================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
  if request.method == "POST":
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    usertype = request.form['usertype']
    securityqn = request.form['securityquestion']
    securityans = request.form['securityanswer']

    if not (username and password and email and usertype and securityqn and securityans):
        flash("Error: No field can be empty")
        return redirect(url_for('register'))
    else:
        username = username.strip()
        password = password.strip()
        email = email.strip()
        usertype = usertype.strip()

    # Returns salted pwd hash in format : method$salt$hashedvalue
    hashed_pwd = generate_password_hash(password, 'sha256')

    new_user = User(username=username, email=email, pwdhash=hashed_pwd, usertype=usertype, securityqn=securityqn, securityans=securityans)
    db.session.add(new_user)

    try:
        db.session.commit()
    except:
        flash("Sorry, Username {u} is already taken. Please choose another username".format(u=username))
        return redirect(url_for('register'))

    flash("Info: User account has been created.")
    return redirect(url_for("login"))

  return render_template("register.html")

# ==================================================================
# Password Reset 
# ==================================================================
@app.route('/passreset', methods=["GET", "POST"])
def passreset():
  return render_template("password_reset_email.html")

@app.route('/reset', methods=["GET", "POST"])
def reset():
  if request.method == "POST":
    form_email = request.form['email']

    if not (form_email):
      flash("Error: Your security field was empty, please try again")
      return redirect(url_for('passreset'))

    try:
      user = User.query.filter_by(email=form_email).first_or_404()
      question = user.securityqn
      return render_template("answer_sec_qn.html", email = form_email, secqn= question)
    except:
      flash('ERROR: Invalid email address!')
      return render_template('password_reset_email.html')
  
@app.route('/reset_ans_question', methods=["GET", "POST"])
def reset_ans_question():
  if request.method == "POST":
    email = request.form['email']
    answer = request.form['answer']

    if not (email and answer):
      flash("Error: Your new password field was empty, please try again")
      return render_template('password_reset_email.html')

    try:
      user = User.query.filter_by(email=email).first_or_404()
      if user.securityans == answer:
        return render_template('change_password.html', email=email)
      else:
        flash('ERROR: Sorry you have input the wrong answer to the security question, please try again')
        user = User.query.filter_by(email=email).first_or_404()
        question = user.securityqn
        return render_template("password_reset_email.html", email = email, secqn= question)
    except:
      flash('ERROR: Invalid email address!')
      return render_template('password_reset_email.html')

@app.route('/change_password', methods=["GET", "POST"])
def change_password():
  if request.method == "POST":
    email = request.form['email']
    password = request.form['password']

    if not (email and password):
      flash("Error: No field can be empty")
      return redirect(url_for('passreset'))

    try:
      user = User.query.filter_by(email=email).first_or_404()
      hashed_pwd = generate_password_hash(password, 'sha256')
      user.pwdhash = hashed_pwd
      db.session.add(user)
      db.session.commit()
      flash('SUCCESS: Your password has been updated!')
      return redirect(url_for('login'))
    except:
      flash('ERROR: Sorry there was an error, please try again')
      return redirect(url_for('passreset'))

# ==================================================================
# Homepages 
# ==================================================================

# Homepage for admin
@app.route('/homepage')
def homepage():
  if 'username' in session:
      username = session['username']
      return render_template("homepage.html", username = username)
  flash("Please login to view this page")
  return redirect("/")

# Homepage for surgeon
@app.route('/userhomepage')
def userhomepage():
  if 'username' in session:
    try:
      username = session['username']
      f = open('/app/files/trainedModels.txt', 'r')
      topline = f.readline()
      f.close()

      topline = str(topline)
      split_topline = topline.split("_")
      comorbidities = split_topline[1]
      comorbidities = comorbidities[1:-1]
      comorbidities_strip = [ x.strip() for x in comorbidities.split(',') ]
      comorbidities_str = [ x[1:-1] for x in comorbidities_strip ]
      comorbidities_array = np.array(comorbidities_str)

      age = comorbidities_array[0]

      # remove age from array
      comorbidities_array_noage = []
      for c in comorbidities_array:
        if c != 'age':
          comorbidities_array_noage.append(c)
  
      comorbidities_array_noage_len = len(comorbidities_array_noage)

      return render_template("userhomepage.html", comorbid = comorbidities_array_noage, length = comorbidities_array_noage_len, username = username)
    except:
      flash('Sorry, no models have been created yet, so no predictions can be made.')
      flash('Please contact your system admin to create models.')
      return redirect("/logout")
  
  flash("Please login to view this page")
  return redirect("/")

# ==================================================================
# Admin section of code - performs admin functions:
# Upload data
# Generate new models
# Regeneration of commonly used models
# Predicting outcome
# ==================================================================

# Upload function
# This code is adapted from the following documentation:
# http://flask.pocoo.org/docs/1.0/patterns/fileuploads/ 
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
  if 'username' in session:
    username = session['username']

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Error: No file uploaded')
            return redirect("/upload")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('Error: No selected file')
            return redirect("/upload")
        if file: 
          if allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "Data.csv"))
            return redirect("/process")
          else:
            flash('Error: Please select either a csv or txt file.')
            return redirect("/upload")
    return render_template("before_upload.html")
  
  flash("Please login to view this page")
  return redirect("/")

# Process the data to make it solely numerical   
@app.route('/process', methods=['GET', 'POST'])
def process():
  if 'username' in session:
    username = session['username']

    if request.method == 'GET':
      try:
        # read datafile 
        data_file = '/app/files/Data.csv'
        processed_data_file = '/app/files/processedData.csv'
      
        with open(data_file, "rt") as fin:
          with open(processed_data_file, "wt") as fout:
              for line in fin:
                fout.write(line.replace('NTGH', '1')
                        .replace('HEXH', '2')
                        .replace('WANS', '3')
                        .replace('NSEC', '4')
                        .replace('W', '0')
                        .replace('HIP', '1')
                        .replace('KNEE', '2')
                        .replace('T', '1')
                        .replace('F', '0')
                        .replace('M', '1')
                        .replace('F', '0')                       
                        .replace('Y', '1')
                        .replace('N', '0')
                        .replace('_', ''))

        # read the processed data file again and replace any missing values with the values of that column
        df = pd.read_csv(processed_data_file)
        df2 = df.fillna(df.median())

        df2.to_csv(processed_data_file, index=False)

        stats = df2.describe()
        stats = np.round(stats,3)
        stats.to_csv('/app/files/summaryStats.csv', index=False)

        try:
          outcomestats = stats[['dvt60', 'pe60', 'pne30', 'ren30', 'dhosp90', 'dout90']]
        except:
          flash('Sorry, the statistics cannot be displayed. Please download the file to view them.')

        return render_template("process.html", stats = outcomestats.to_html())
      except:
        flash('Error: There is no datafile to process. Please upload a data file first')
        return redirect('/upload')
  
  flash("Please login to view this page")
  return redirect("/")

# Export Summary Stats in csv
@app.route('/exportStats')
def exportStats():
  if 'username' in session:
      username = session['username']

      try:
        path = "summaryStats.csv"
        return send_from_directory(UPLOAD_FOLDER, path, as_attachment=True)
      except:
        flash('Error: There is no datafile to export. Please upload data first')
        return redirect('/process')
  
  flash("Please login to view this page")
  return redirect("/")

# Page for selecting columns as predictors and outcomes
@app.route('/selectcols', methods=['GET', 'POST'])
def selectcols():
  if 'username' in session:
    username = session['username']

    try:
      full_path_to_file2 = "/app/files/processedData.csv"

      data = pd.read_csv(full_path_to_file2, nrows=1)
      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      no_columns = len(headers_array)

      path = "trainedModels.txt"
      file1 = open(os.path.join(app.config['UPLOAD_FOLDER'], path), 'r')

      #file1 = open('/app/files/trainedmodels.txt', 'r')
      modelnames = file1.readlines()
      names = []
      for model in modelnames:
        model_array = model.split("_")
        names.append(model_array[0])
      modelnames_length = len(modelnames)

      file1.close()

      return render_template("selectcols.html", headers = headers_array, no_columns = no_columns, modelnames = names, modelnames_length = modelnames_length)
    except:
      flash('Error: There is no datafile uploaded. Please upload a data file first')
      return redirect('/upload')

  flash("Please login to view this page")
  return redirect("/")

# General method to create a Random Forest model 
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/buildmodel', methods=['GET', 'POST'])
def buildmodel():
  if 'username' in session:
      username = session['username']

      values = request.form 

      predictors = values['predictors']
      outcome = values['outcome']
      modelname = values['modelName']

      # check values were given for every field 
      if not (predictors and outcome and modelname):
        flash("Error: No field can be empty.")
        return redirect(url_for('selectcols'))
      else:
        predictors = predictors.strip()
        outcome = outcome.strip()
        modelname = modelname.strip()

      try:
        # load the file, seperate into train and test data, then run the algorithm

        full_path_to_file = '/app/files/processedData.csv'

        data = pd.read_csv(full_path_to_file)

        # The inputs 
        predColumns = predictors.split(',')
        predColumnsInt = [ int(x) for x in predColumns ]
        predColumnsInt2 = [ x-1 for x in predColumnsInt ]
        predColumnsInt3 = list(predColumnsInt2)
  
        # The target
        outcome = int(outcome)
        outcome = outcome - 1 

        # Get target column name 
        headers_list = list(data.columns.values)
        headers_array = np.asarray(headers_list)
        names_outcome_col = ""
        names_outcome_col += headers_array[outcome]

        # Separate majority and minority classes
        df_majority = data[data[names_outcome_col]==0]
        df_minority = data[data[names_outcome_col]==1]
  
        # Check value counts
        # Value count of 0 will be the upsample number
        value_count_0 = data[names_outcome_col].value_counts()[0]
        value_count_1 = data[names_outcome_col].value_counts()[1]

        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                  replace=True,     # sample with replacement
                                  n_samples=value_count_0,    # to match majority class
                                  random_state=123) # reproducible results
  
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])

        # Up-sampled inputs and targets
        data_inputs =  df_upsampled.iloc[:, np.r_[predColumnsInt3]]
        data_targets = df_upsampled.iloc[:,outcome] 

        # Not up-sampled inputs and targets
        data_inputs2 = data.iloc[:, np.r_[predColumnsInt3]]
        data_targets2 = data.iloc[:,outcome] 

        X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

        clf = RandomForestClassifier(n_estimators=100)
        name = modelname

        roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
        roc = round(roc, 3)
        roc = str(roc) 

        precision1 = round(class_report[0][0], 3)
        precision2 = round(class_report[0][1], 3)
        recall1 = round(class_report[1][0],3)
        recall2 = round(class_report[1][1],3)

        # calculate average risk and standard dev to be printed with predictions
        total_column = value_count_0 + value_count_1
        average_risk = float(value_count_1) / float(total_column)
        average_risk = average_risk*100
        average_risk = round(average_risk,3)

        std = data[names_outcome_col].std()
        av_plus_std = average_risk + (2.0 * std)
        av_plus_std = round(av_plus_std,3)
        
        # write inputs and targets to file so we know the model trained
        # get the names of the input and target cols instead of using indices
        names_pred_cols = []

        for p in predColumnsInt2:
          name = headers_array[p]
          names_pred_cols.append(name)    

        # delete the line and add new line
        delete_line = str(modelname)
        f = open('/app/files/trainedModels.txt', 'r')
        lines = f.readlines()
        f.close()
        f = open("/app/files/trainedModels.txt","w")
        for line in lines:
          if delete_line not in line:
            f.write(line)
        #f.close()

        # add new line
        #f = open("/app/files/trainedModels.txt","a")
        new_line = str(modelname)
        new_line += "_"
        new_line += str(names_pred_cols)
        new_line += "_"
        new_line += str(names_outcome_col)
        new_line += "_"
        new_line += str(confusionmtx[1][1])
        new_line += "_"
        new_line += str(confusionmtx[0][1])
        new_line += "_"
        new_line += str(confusionmtx[0][0])
        new_line += "_"
        new_line += str(confusionmtx[1][0])
        new_line += "_"
        new_line += str(roc)
        new_line += "_"
        new_line += str(precision1)
        new_line += "_"
        new_line += str(precision2)
        new_line += "_"
        new_line += str(recall1)
        new_line += "_"
        new_line += str(recall2)
        new_line += "_"
        new_line += str(value_count_0)
        new_line += "_"
        new_line += str(value_count_1)
        new_line += "_"
        new_line += str(average_risk)
        new_line += "_"
        new_line += str(av_plus_std)
        new_line += '\n'
        f.write(new_line)
        f.close()

        return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )
      except:
        flash('Error: There is no datafile uploaded. Please upload a data file first')
        return redirect('/upload')
  
  flash("Please login to view this page")
  return redirect("/")

# ==================================================================
# Subsequent 8 methods are to regenerate common models 
# with one click 
# Models: renal, dvt, pe, dhosp90, dout90, pne, mi and readm
# ==================================================================

# Regenerate renal model
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/modelrenal', methods=['GET', 'POST'])
def modelrenal():
  if 'username' in session:
    username = session['username']
    
    try:
      # load the file, seperate into train and test data, then run the algorithm

      full_path_to_file = '/app/files/processedData.csv'

      data = pd.read_csv(full_path_to_file)

      # load in columns from file

      l = "ren30"
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      predictorsStr = np.array(predictorsStr)
      
      outcome = split_trained_line[2]
      outcome = outcome.rstrip()
      outcome = str(outcome)

      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      predictors_columns = []
      #outcome_column = []

      for p in range(len(predictorsStr)):
        for h in range(len(headers_array)):
            if headers_array[h] == predictorsStr[p]:
                predictors_columns.append(h)
                break

      #outcome_column = 0
      for header in range(len(headers_array)):
        if headers_array[header] == outcome:
          outcome_column = header
          #break

      # Separate majority and minority classes
      df_majority = data[data[outcome]==0]
      df_minority = data[data[outcome]==1]
 
      # Check value counts
      # Value count of 0 will be the upsample number
      value_count_0 = data[outcome].value_counts()[0]
      value_count_1 = data[outcome].value_counts()[1]

      # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=value_count_0,    # to match majority class
                                 random_state=123) # reproducible results
 
      # Combine majority class with upsampled minority class
      df_upsampled = pd.concat([df_majority, df_minority_upsampled])

      # Up-sampled inputs and targets
      data_inputs =  df_upsampled.iloc[:, np.r_[predictors_columns]]
      data_targets = df_upsampled.iloc[:,outcome_column] 

      # Not up-sampled inputs and targets
      data_inputs2 = data.iloc[:, np.r_[predictors_columns]]
      data_targets2 = data.iloc[:,outcome_column] 

      X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

      clf = RandomForestClassifier(n_estimators=100)
      name = 'ren30'

      roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
      roc = round(roc, 3)
      roc = str(roc) 

      precision1 = round(class_report[0][0], 3)
      precision2 = round(class_report[0][1], 3)
      recall1 = round(class_report[1][0],3)
      recall2 = round(class_report[1][1],3)

      # calculate average risk to be printed with predictions
      total_column = value_count_0 + value_count_1
      average_risk = float(value_count_1) / float(total_column)
      average_risk = average_risk*100
      average_risk = round(average_risk,3)

      std = data[outcome].std()
      av_plus_std = average_risk + (2.0 * std)
      av_plus_std = round(av_plus_std,3)

      # update the line
      
      # delete the old line 
      delete_line = str(trained_line)
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      f.close()
      f = open("/app/files/trainedModels.txt","w")
      for line in lines:
        if delete_line not in line:
          f.write(line)
      

      # add in a new line
      new_line = str(split_trained_line[0])
      new_line += "_"
      new_line += str(split_trained_line[1])
      new_line += "_"
      new_line += str(split_trained_line[2])
      new_line += "_"
      new_line += str(confusionmtx[1][1])
      new_line += "_"
      new_line += str(confusionmtx[0][1])
      new_line += "_"
      new_line += str(confusionmtx[0][0])
      new_line += "_"
      new_line += str(confusionmtx[1][0])
      new_line += "_"
      new_line += str(roc)
      new_line += "_"
      new_line += str(precision1)
      new_line += "_"
      new_line += str(precision2)
      new_line += "_"
      new_line += str(recall1)
      new_line += "_"
      new_line += str(recall2)
      new_line += "_"
      new_line += str(split_trained_line[12])
      new_line += "_"
      new_line += str(split_trained_line[13].strip())
      new_line += "_"
      new_line += str(average_risk)
      new_line += "_"
      new_line += str(av_plus_std)
      new_line += '\n'
      f.write(new_line)
      f.close()

      #return str(df_upsampled['ren30'].std())
      return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )
    except:
      flash('Error: There is no model generated. Please generate a model first')
      return redirect('/selectcols')
  
  flash("Please login to view this page")
  return redirect("/")

# Regenerate dvt model
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/modeldvt', methods=['GET', 'POST'])
def modeldvt():
  if 'username' in session:
    username = session['username']
    # load the file, seperate into train and test data, then run the algorithm

    try:
      full_path_to_file = '/app/files/processedData.csv'

      data = pd.read_csv(full_path_to_file)

      # load in columns from file

      l = "dvt60"
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      predictorsStr = np.array(predictorsStr)
      
      outcome = split_trained_line[2]
      outcome = outcome.rstrip()
      outcome = str(outcome)

      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      predictors_columns = []
      #outcome_column = []

      for p in range(len(predictorsStr)):
        for h in range(len(headers_array)):
            if headers_array[h] == predictorsStr[p]:
                predictors_columns.append(h)
                break

      #outcome_column = 0
      for header in range(len(headers_array)):
        if headers_array[header] == outcome:
          outcome_column = header
          #break

      # Separate majority and minority classes
      df_majority = data[data[outcome]==0]
      df_minority = data[data[outcome]==1]
 
      # Check value counts
      # Value count of 0 will be the upsample number
      value_count_0 = data[outcome].value_counts()[0]
      value_count_1 = data[outcome].value_counts()[1]

      # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=value_count_0,    # to match majority class
                                 random_state=123) # reproducible results
 
      # Combine majority class with upsampled minority class
      df_upsampled = pd.concat([df_majority, df_minority_upsampled])

      # Up-sampled inputs and targets
      data_inputs =  df_upsampled.iloc[:, np.r_[predictors_columns]]
      data_targets = df_upsampled.iloc[:,outcome_column] 

      # Not up-sampled inputs and targets
      data_inputs2 = data.iloc[:, np.r_[predictors_columns]]
      data_targets2 = data.iloc[:,outcome_column] 

      X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

      clf = RandomForestClassifier(n_estimators=100)
      name = 'dvt60'

      roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
      roc = round(roc, 3)
      roc = str(roc) 

      precision1 = round(class_report[0][0], 3)
      precision2 = round(class_report[0][1], 3)
      recall1 = round(class_report[1][0],3)
      recall2 = round(class_report[1][1],3)

      # calculate average risk to be printed with predictions
      total_column = value_count_0 + value_count_1
      average_risk = float(value_count_1) / float(total_column)
      average_risk = average_risk*100
      average_risk = round(average_risk,3)
      
      std = data[outcome].std()
      std_times_two = std * 2.0
      av_plus_std = average_risk + std_times_two
      av_plus_std = round(av_plus_std,3)

      # update the line
      
      # delete the old line 
      delete_line = str(trained_line)
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      f.close()
      f = open("/app/files/trainedModels.txt","w")
      for line in lines:
        if delete_line not in line:
          f.write(line)

      # add in a new line
      new_line = str(split_trained_line[0])
      new_line += "_"
      new_line += str(split_trained_line[1])
      new_line += "_"
      new_line += str(split_trained_line[2])
      new_line += "_"
      new_line += str(confusionmtx[1][1])
      new_line += "_"
      new_line += str(confusionmtx[0][1])
      new_line += "_"
      new_line += str(confusionmtx[0][0])
      new_line += "_"
      new_line += str(confusionmtx[1][0])
      new_line += "_"
      new_line += str(roc)
      new_line += "_"
      new_line += str(precision1)
      new_line += "_"
      new_line += str(precision2)
      new_line += "_"
      new_line += str(recall1)
      new_line += "_"
      new_line += str(recall2)
      new_line += "_"
      new_line += str(split_trained_line[12])
      new_line += "_"
      new_line += str(split_trained_line[13].strip())
      new_line += "_"
      new_line += str(average_risk)
      new_line += "_"
      new_line += str(av_plus_std)
      new_line += '\n'
      f.write(new_line)
      f.close()
      
      return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )
      #return str(data['ren30'].std())

    except:
      flash('Error: There is no model generated. Please generate a model first')
      return redirect('/selectcols')

  flash("Please login to view this page")
  return redirect("/")      

# Regenerate pe model
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/modelpe', methods=['GET', 'POST'])
def modelpe():
  if 'username' in session:
    username = session['username']

    try:
      # load the file, seperate into train and test data, then run the algorithm

      full_path_to_file = '/app/files/processedData.csv'

      data = pd.read_csv(full_path_to_file)

      # load in columns from file

      l = "pe60"
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      predictorsStr = np.array(predictorsStr)
      
      outcome = split_trained_line[2]
      outcome = outcome.rstrip()
      outcome = str(outcome)

      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      predictors_columns = []
      #outcome_column = []

      for p in range(len(predictorsStr)):
        for h in range(len(headers_array)):
            if headers_array[h] == predictorsStr[p]:
                predictors_columns.append(h)
                break

      #outcome_column = 0
      for header in range(len(headers_array)):
        if headers_array[header] == outcome:
          outcome_column = header
          #break

      # Separate majority and minority classes
      df_majority = data[data[outcome]==0]
      df_minority = data[data[outcome]==1]
 
      # Check value counts
      # Value count of 0 will be the upsample number
      value_count_0 = data[outcome].value_counts()[0]
      value_count_1 = data[outcome].value_counts()[1]

      # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=value_count_0,    # to match majority class
                                 random_state=123) # reproducible results
 
      # Combine majority class with upsampled minority class
      df_upsampled = pd.concat([df_majority, df_minority_upsampled])

      # Up-sampled inputs and targets
      data_inputs =  df_upsampled.iloc[:, np.r_[predictors_columns]]
      data_targets = df_upsampled.iloc[:,outcome_column] 

      # Not up-sampled inputs and targets
      data_inputs2 = data.iloc[:, np.r_[predictors_columns]]
      data_targets2 = data.iloc[:,outcome_column] 

      X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

      clf = RandomForestClassifier(n_estimators=100)
      name = 'pe60'

      roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
      roc = round(roc, 3)
      roc = str(roc) 

      precision1 = round(class_report[0][0], 3)
      precision2 = round(class_report[0][1], 3)
      recall1 = round(class_report[1][0],3)
      recall2 = round(class_report[1][1],3)

      # calculate average risk to be printed with predictions
      total_column = value_count_0 + value_count_1
      average_risk = float(value_count_1) / float(total_column)
      average_risk = average_risk*100
      average_risk = round(average_risk,3)
      std = data[outcome].std()
      av_plus_std = average_risk + (2.0 * std)
      av_plus_std = round(av_plus_std,3)

      # update the line
      
      # delete the old line 
      delete_line = str(trained_line)
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      f.close()
      f = open("/app/files/trainedModels.txt","w")
      for line in lines:
        if delete_line not in line:
          f.write(line)

      # add in a new line
      new_line = str(split_trained_line[0])
      new_line += "_"
      new_line += str(split_trained_line[1])
      new_line += "_"
      new_line += str(split_trained_line[2])
      new_line += "_"
      new_line += str(confusionmtx[1][1])
      new_line += "_"
      new_line += str(confusionmtx[0][1])
      new_line += "_"
      new_line += str(confusionmtx[0][0])
      new_line += "_"
      new_line += str(confusionmtx[1][0])
      new_line += "_"
      new_line += str(roc)
      new_line += "_"
      new_line += str(precision1)
      new_line += "_"
      new_line += str(precision2)
      new_line += "_"
      new_line += str(recall1)
      new_line += "_"
      new_line += str(recall2)
      new_line += "_"
      new_line += str(split_trained_line[12])
      new_line += "_"
      new_line += str(split_trained_line[13].strip())
      new_line += "_"
      new_line += str(average_risk)
      new_line += "_"
      new_line += str(av_plus_std)
      new_line += '\n'
      f.write(new_line)
      f.close()
      
      return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )
    
    except:
      flash('Error: There is no model generated. Please generate a model first')
      return redirect('/selectcols')
  
  flash("Please login to view this page")
  return redirect("/")

# Regenerate dhosp90 model
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/modeldhosp90', methods=['GET', 'POST'])
def modeldhosp90():
  if 'username' in session:
    username = session['username']  

    try:
      # load the file, seperate into train and test data, then run the algorithm

      full_path_to_file = '/app/files/processedData.csv'

      data = pd.read_csv(full_path_to_file)

      # load in columns from file

      l = "dhosp90"
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      predictorsStr = np.array(predictorsStr)
      
      outcome = split_trained_line[2]
      outcome = outcome.rstrip()
      outcome = str(outcome)

      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      predictors_columns = []
      #outcome_column = []

      for p in range(len(predictorsStr)):
        for h in range(len(headers_array)):
            if headers_array[h] == predictorsStr[p]:
                predictors_columns.append(h)
                break

      #outcome_column = 0
      for header in range(len(headers_array)):
        if headers_array[header] == outcome:
          outcome_column = header
          #break

      # Separate majority and minority classes
      df_majority = data[data[outcome]==0]
      df_minority = data[data[outcome]==1]
 
      # Check value counts
      # Value count of 0 will be the upsample number
      value_count_0 = data[outcome].value_counts()[0]
      value_count_1 = data[outcome].value_counts()[1]

      # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=value_count_0,    # to match majority class
                                 random_state=123) # reproducible results
 
      # Combine majority class with upsampled minority class
      df_upsampled = pd.concat([df_majority, df_minority_upsampled])

      # Up-sampled inputs and targets
      data_inputs =  df_upsampled.iloc[:, np.r_[predictors_columns]]
      data_targets = df_upsampled.iloc[:,outcome_column] 

      # Not up-sampled inputs and targets
      data_inputs2 = data.iloc[:, np.r_[predictors_columns]]
      data_targets2 = data.iloc[:,outcome_column] 

      X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

      clf = RandomForestClassifier(n_estimators=100)
      name = 'dhosp90'

      roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
      roc = round(roc, 3)
      roc = str(roc) 

      precision1 = round(class_report[0][0], 3)
      precision2 = round(class_report[0][1], 3)
      recall1 = round(class_report[1][0],3)
      recall2 = round(class_report[1][1],3)


      # calculate average risk to be printed with predictions
      total_column = value_count_0 + value_count_1
      average_risk = float(value_count_1) / float(total_column)
      average_risk = average_risk*100
      average_risk = round(average_risk,3)
      std = data[outcome].std()
      av_plus_std = average_risk + (2.0 * std)
      av_plus_std = round(av_plus_std,3)

      # update the line
      
      # delete the old line 
      delete_line = str(trained_line)
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      f.close()
      f = open("/app/files/trainedModels.txt","w")
      for line in lines:
        if delete_line not in line:
          f.write(line)

      # add in a new line
      new_line = str(split_trained_line[0])
      new_line += "_"
      new_line += str(split_trained_line[1])
      new_line += "_"
      new_line += str(split_trained_line[2])
      new_line += "_"
      new_line += str(confusionmtx[1][1])
      new_line += "_"
      new_line += str(confusionmtx[0][1])
      new_line += "_"
      new_line += str(confusionmtx[0][0])
      new_line += "_"
      new_line += str(confusionmtx[1][0])
      new_line += "_"
      new_line += str(roc)
      new_line += "_"
      new_line += str(precision1)
      new_line += "_"
      new_line += str(precision2)
      new_line += "_"
      new_line += str(recall1)
      new_line += "_"
      new_line += str(recall2)
      new_line += "_"
      new_line += str(split_trained_line[12])
      new_line += "_"
      new_line += str(split_trained_line[13].strip())
      new_line += "_"
      new_line += str(average_risk)
      new_line += "_"
      new_line += str(av_plus_std)
      new_line += '\n'
      f.write(new_line)
      f.close()
      
      return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )

    except:
      flash('Error: There is no model generated. Please generate a model first')
      return redirect('/selectcols')
  
  flash("Please login to view this page")
  return redirect("/")

# Regenerate dout90 model
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/modeldout90', methods=['GET', 'POST'])
def modeldout90():
  if 'username' in session:
    username = session['username']

    try:
      # load the file, seperate into train and test data, then run the algorithm

      full_path_to_file = '/app/files/processedData.csv'

      data = pd.read_csv(full_path_to_file)

      # load in columns from file

      l = "dout90"
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      predictorsStr = np.array(predictorsStr)
      
      outcome = split_trained_line[2]
      outcome = outcome.rstrip()
      outcome = str(outcome)

      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      predictors_columns = []
      #outcome_column = []

      for p in range(len(predictorsStr)):
        for h in range(len(headers_array)):
            if headers_array[h] == predictorsStr[p]:
                predictors_columns.append(h)
                break

      #outcome_column = 0
      for header in range(len(headers_array)):
        if headers_array[header] == outcome:
          outcome_column = header
          #break

      # Separate majority and minority classes
      df_majority = data[data[outcome]==0]
      df_minority = data[data[outcome]==1]
 
      # Check value counts
      # Value count of 0 will be the upsample number
      value_count_0 = data[outcome].value_counts()[0]
      value_count_1 = data[outcome].value_counts()[1]

      # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=value_count_0,    # to match majority class
                                 random_state=123) # reproducible results
 
      # Combine majority class with upsampled minority class
      df_upsampled = pd.concat([df_majority, df_minority_upsampled])

      # Up-sampled inputs and targets
      data_inputs =  df_upsampled.iloc[:, np.r_[predictors_columns]]
      data_targets = df_upsampled.iloc[:,outcome_column] 

      # Not up-sampled inputs and targets
      data_inputs2 = data.iloc[:, np.r_[predictors_columns]]
      data_targets2 = data.iloc[:,outcome_column] 

      X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

      clf = RandomForestClassifier(n_estimators=100)
      name = 'dout90'

      roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
      roc = round(roc, 3)
      roc = str(roc) 

      precision1 = round(class_report[0][0], 3)
      precision2 = round(class_report[0][1], 3)
      recall1 = round(class_report[1][0],3)
      recall2 = round(class_report[1][1],3)

      # calculate average risk to be printed with predictions
      total_column = value_count_0 + value_count_1
      average_risk = float(value_count_1) / float(total_column)
      average_risk = average_risk*100
      average_risk = round(average_risk,3)
      std = data[outcome].std()
      av_plus_std = average_risk + (2.0 * std)
      av_plus_std = round(av_plus_std,3)

      # update the line
      
      # delete the old line 
      delete_line = str(trained_line)
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      f.close()
      f = open("/app/files/trainedModels.txt","w")
      for line in lines:
        if delete_line not in line:
          f.write(line)

      # add in a new line
      new_line = str(split_trained_line[0])
      new_line += "_"
      new_line += str(split_trained_line[1])
      new_line += "_"
      new_line += str(split_trained_line[2])
      new_line += "_"
      new_line += str(confusionmtx[1][1])
      new_line += "_"
      new_line += str(confusionmtx[0][1])
      new_line += "_"
      new_line += str(confusionmtx[0][0])
      new_line += "_"
      new_line += str(confusionmtx[1][0])
      new_line += "_"
      new_line += str(roc)
      new_line += "_"
      new_line += str(precision1)
      new_line += "_"
      new_line += str(precision2)
      new_line += "_"
      new_line += str(recall1)
      new_line += "_"
      new_line += str(recall2)
      new_line += "_"
      new_line += str(split_trained_line[12])
      new_line += "_"
      new_line += str(split_trained_line[13].strip())
      new_line += "_"
      new_line += str(average_risk)
      new_line += "_"
      new_line += str(av_plus_std)
      new_line += '\n'
      f.write(new_line)
      f.close()
      
      return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )
    
    except:
      flash('Error: There is no model generated. Please generate a model first')
      return redirect('/selectcols')
  
  flash("Please login to view this page")
  return redirect("/")

# Regenerate pne model
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/modelpne', methods=['GET', 'POST'])
def modelpne():
  if 'username' in session:
    username = session['username']

    try:
      # load the file, seperate into train and test data, then run the algorithm

      full_path_to_file = '/app/files/processedData.csv'

      data = pd.read_csv(full_path_to_file)

      # load in columns from file

      l = "pne30"
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      predictorsStr = np.array(predictorsStr)
      
      outcome = split_trained_line[2]
      outcome = outcome.rstrip()
      outcome = str(outcome)

      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      predictors_columns = []
      #outcome_column = []

      for p in range(len(predictorsStr)):
        for h in range(len(headers_array)):
            if headers_array[h] == predictorsStr[p]:
                predictors_columns.append(h)
                break

      #outcome_column = 0
      for header in range(len(headers_array)):
        if headers_array[header] == outcome:
          outcome_column = header
          #break

      # Separate majority and minority classes
      df_majority = data[data[outcome]==0]
      df_minority = data[data[outcome]==1]
 
      # Check value counts
      # Value count of 0 will be the upsample number
      value_count_0 = data[outcome].value_counts()[0]
      value_count_1 = data[outcome].value_counts()[1]

      # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=value_count_0,    # to match majority class
                                 random_state=123) # reproducible results
 
      # Combine majority class with upsampled minority class
      df_upsampled = pd.concat([df_majority, df_minority_upsampled])

      # Up-sampled inputs and targets
      data_inputs =  df_upsampled.iloc[:, np.r_[predictors_columns]]
      data_targets = df_upsampled.iloc[:,outcome_column] 

      # Not up-sampled inputs and targets
      data_inputs2 = data.iloc[:, np.r_[predictors_columns]]
      data_targets2 = data.iloc[:,outcome_column] 

      X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

      clf = RandomForestClassifier(n_estimators=100)
      name = 'pne30'

      roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
      roc = round(roc, 3)
      roc = str(roc) 

      precision1 = round(class_report[0][0], 3)
      precision2 = round(class_report[0][1], 3)
      recall1 = round(class_report[1][0],3)
      recall2 = round(class_report[1][1],3)

      # calculate average risk to be printed with predictions
      total_column = value_count_0 + value_count_1
      average_risk = float(value_count_1) / float(total_column)
      average_risk = average_risk*100
      average_risk = round(average_risk,3)
      std = data[outcome].std()
      av_plus_std = average_risk + (2.0 * std)
      av_plus_std = round(av_plus_std,3)

      # update the line

      # append the false negative and false positive rates to be seen with predictions
      
      # delete the old line 
      delete_line = str(trained_line)
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      f.close()
      f = open("/app/files/trainedModels.txt","w")
      for line in lines:
        if delete_line not in line:
          f.write(line)

      # add in a new line
      new_line = str(split_trained_line[0])
      new_line += "_"
      new_line += str(split_trained_line[1])
      new_line += "_"
      new_line += str(split_trained_line[2])
      new_line += "_"
      new_line += str(confusionmtx[1][1])
      new_line += "_"
      new_line += str(confusionmtx[0][1])
      new_line += "_"
      new_line += str(confusionmtx[0][0])
      new_line += "_"
      new_line += str(confusionmtx[1][0])
      new_line += "_"
      new_line += str(roc)
      new_line += "_"
      new_line += str(precision1)
      new_line += "_"
      new_line += str(precision2)
      new_line += "_"
      new_line += str(recall1)
      new_line += "_"
      new_line += str(recall2)
      new_line += "_"
      new_line += str(split_trained_line[12])
      new_line += "_"
      new_line += str(split_trained_line[13].strip())
      new_line += "_"
      new_line += str(average_risk)
      new_line += "_"
      new_line += str(av_plus_std)
      new_line += '\n'
      f.write(new_line)
      f.close()
      
      return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )
    
    except:
      flash('Error: There is no model generated. Please generate a model first')
      return redirect('/selectcols')
  
  flash("Please login to view this page")
  return redirect("/")

# Regenerate mi model
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/modelmi', methods=['GET', 'POST'])
def modelmi():
  if 'username' in session:
    username = session['username']

    try:
      # load the file, seperate into train and test data, then run the algorithm

      full_path_to_file = '/app/files/processedData.csv'

      data = pd.read_csv(full_path_to_file)

      # load in columns from file

      l = "mi30"
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      predictorsStr = np.array(predictorsStr)
      
      outcome = split_trained_line[2]
      outcome = outcome.rstrip()
      outcome = str(outcome)

      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      predictors_columns = []
      #outcome_column = []

      for p in range(len(predictorsStr)):
        for h in range(len(headers_array)):
            if headers_array[h] == predictorsStr[p]:
                predictors_columns.append(h)
                break

      #outcome_column = 0
      for header in range(len(headers_array)):
        if headers_array[header] == outcome:
          outcome_column = header
          #break

      # Separate majority and minority classes
      df_majority = data[data[outcome]==0]
      df_minority = data[data[outcome]==1]
 
      # Check value counts
      # Value count of 0 will be the upsample number
      value_count_0 = data[outcome].value_counts()[0]
      value_count_1 = data[outcome].value_counts()[1]

      # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=value_count_0,    # to match majority class
                                 random_state=123) # reproducible results
 
      # Combine majority class with upsampled minority class
      df_upsampled = pd.concat([df_majority, df_minority_upsampled])

      # Up-sampled inputs and targets
      data_inputs =  df_upsampled.iloc[:, np.r_[predictors_columns]]
      data_targets = df_upsampled.iloc[:,outcome_column] 

      # Not up-sampled inputs and targets
      data_inputs2 = data.iloc[:, np.r_[predictors_columns]]
      data_targets2 = data.iloc[:,outcome_column] 

      X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

      clf = RandomForestClassifier(n_estimators=100)
      name = 'mi30'

      roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
      roc = round(roc, 3)
      roc = str(roc) 

      precision1 = round(class_report[0][0], 3)
      precision2 = round(class_report[0][1], 3)
      recall1 = round(class_report[1][0],3)
      recall2 = round(class_report[1][1],3)

      # calculate average risk to be printed with predictions
      total_column = value_count_0 + value_count_1
      average_risk = float(value_count_1) / float(total_column)
      average_risk = average_risk*100
      average_risk = round(average_risk,3)
      std = data[outcome].std()
      av_plus_std = average_risk + (2.0 * std)
      av_plus_std = round(av_plus_std,3)

      # update the line
      
      # delete the old line 
      delete_line = str(trained_line)
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      f.close()
      f = open("/app/files/trainedModels.txt","w")
      for line in lines:
        if delete_line not in line:
          f.write(line)

      # add in a new line
      new_line = str(split_trained_line[0])
      new_line += "_"
      new_line += str(split_trained_line[1])
      new_line += "_"
      new_line += str(split_trained_line[2])
      new_line += "_"
      new_line += str(confusionmtx[1][1])
      new_line += "_"
      new_line += str(confusionmtx[0][1])
      new_line += "_"
      new_line += str(confusionmtx[0][0])
      new_line += "_"
      new_line += str(confusionmtx[1][0])
      new_line += "_"
      new_line += str(roc)
      new_line += "_"
      new_line += str(precision1)
      new_line += "_"
      new_line += str(precision2)
      new_line += "_"
      new_line += str(recall1)
      new_line += "_"
      new_line += str(recall2)
      new_line += "_"
      new_line += str(split_trained_line[12])
      new_line += "_"
      new_line += str(split_trained_line[13].strip())
      new_line += "_"
      new_line += str(average_risk)
      new_line += "_"
      new_line += str(av_plus_std)
      new_line += '\n'
      f.write(new_line)
      f.close()
      
      return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )
    
    except:
      flash('Error: There is no model generated. Please generate a model first')
      return redirect('/selectcols')
  
  flash("Please login to view this page")
  return redirect("/")

# Regenerate readm model
# The upsampling method used here is adapted from the following tutorial: https://elitedatascience.com/imbalanced-classes
@app.route('/modelreadm', methods=['GET', 'POST'])
def modelreadm():
  if 'username' in session:
    username = session['username']

    try:
      # load the file, seperate into train and test data, then run the algorithm

      full_path_to_file = '/app/files/processedData.csv'

      data = pd.read_csv(full_path_to_file)

      # load in columns from file

      l = "readm"
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      predictorsStr = np.array(predictorsStr)
      
      outcome = split_trained_line[2]
      outcome = outcome.rstrip()
      outcome = str(outcome)

      headers_list = list(data.columns.values)
      headers_array = np.asarray(headers_list)

      predictors_columns = []
      #outcome_column = []

      for p in range(len(predictorsStr)):
        for h in range(len(headers_array)):
            if headers_array[h] == predictorsStr[p]:
                predictors_columns.append(h)
                break

      #outcome_column = 0
      for header in range(len(headers_array)):
        if headers_array[header] == outcome:
          outcome_column = header
          #break

      # Separate majority and minority classes
      df_majority = data[data[outcome]==0]
      df_minority = data[data[outcome]==1]
 
      # Check value counts
      # Value count of 0 will be the upsample number
      value_count_0 = data[outcome].value_counts()[0]
      value_count_1 = data[outcome].value_counts()[1]

      # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=value_count_0,    # to match majority class
                                 random_state=123) # reproducible results
 
      # Combine majority class with upsampled minority class
      df_upsampled = pd.concat([df_majority, df_minority_upsampled])

      # Up-sampled inputs and targets
      data_inputs =  df_upsampled.iloc[:, np.r_[predictors_columns]]
      data_targets = df_upsampled.iloc[:,outcome_column] 

      # Not up-sampled inputs and targets
      data_inputs2 = data.iloc[:, np.r_[predictors_columns]]
      data_targets2 = data.iloc[:,outcome_column] 

      X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_targets)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs2, data_targets2)

      clf = RandomForestClassifier(n_estimators=100)
      name = 'readm'

      roc, class_report, confusionmtx = algorithm(clf, name, X_train, X_test, y_train, y_test, X_test2, y_test2)
      roc = round(roc, 3)
      roc = str(roc) 

      precision1 = round(class_report[0][0], 3)
      precision2 = round(class_report[0][1], 3)
      recall1 = round(class_report[1][0],3)
      recall2 = round(class_report[1][1],3)

      # calculate average risk to be printed with predictions
      total_column = value_count_0 + value_count_1
      average_risk = float(value_count_1) / float(total_column)
      average_risk = average_risk*100
      average_risk = round(average_risk,3)
      std = data[outcome].std()
      av_plus_std = average_risk + (2.0 * std)
      av_plus_std = round(av_plus_std,3)

      # update the line
      
      # delete the old line 
      delete_line = str(trained_line)
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      f.close()
      f = open("/app/files/trainedModels.txt","w")
      for line in lines:
        if delete_line not in line:
          f.write(line)

      # add in a new line
      new_line = str(split_trained_line[0])
      new_line += "_"
      new_line += str(split_trained_line[1])
      new_line += "_"
      new_line += str(split_trained_line[2])
      new_line += "_"
      new_line += str(confusionmtx[1][1])
      new_line += "_"
      new_line += str(confusionmtx[0][1])
      new_line += "_"
      new_line += str(confusionmtx[0][0])
      new_line += "_"
      new_line += str(confusionmtx[1][0])
      new_line += "_"
      new_line += str(roc)
      new_line += "_"
      new_line += str(precision1)
      new_line += "_"
      new_line += str(precision2)
      new_line += "_"
      new_line += str(recall1)
      new_line += "_"
      new_line += str(recall2)
      new_line += "_"
      new_line += str(split_trained_line[12])
      new_line += "_"
      new_line += str(split_trained_line[13].strip())
      new_line += "_"
      new_line += str(average_risk)
      new_line += "_"
      new_line += str(av_plus_std)
      new_line += '\n'
      f.write(new_line)
      f.close()
      
      return render_template('trainingSuccessful.html', roc = roc, confusionmtx=confusionmtx, class_report = class_report, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2 )
    
    except:
      flash('Error: There is no model generated. Please generate a model first')
      return redirect('/selectcols')
  
  flash("Please login to view this page")
  return redirect("/")

# ================================================================== 
# Admin Predict 
# ==================================================================

# Method to generate page to choose model to predict from
@app.route('/choosemodel', methods=['GET', 'POST'])
def choosemodel():
  if 'username' in session:
    username = session['username']
    try:

      path = "trainedModels.txt"
      # file1 = open('/app/files/trainedmodels.txt', 'r')
      file1 = open(os.path.join(app.config['UPLOAD_FOLDER'], path), 'r')
      
      modelnames = file1.readlines()
      file1.close()

      if not modelnames:
        flash('Error: Hi, There are no models trained yet. Please train a model first')
        return redirect('/selectcols')
      
      names = []
      for model in modelnames:
        model_array = model.split("_")
        names.append(model_array[0])
      
      return render_template("choosemodel.html", modelnames = names)
    except:
      flash('Error: There are no models trained yet. Please train a model first')
      return redirect('/selectcols')
  
  flash("Please login to view this page")
  return redirect("/")

# Method to generate a page to input values for predicting an outcome using Random Forest
@app.route('/predictmodel', methods=['GET', 'POST'])
def predictmodel():
  if 'username' in session:
    username = session['username']

    try:
      values = request.form 
      modelname = values['modelnames']
      modelname = str(modelname)

      if not (modelname):
        flash("Modelname cannot be empty.")
        return redirect(url_for('choosemodel'))

      # read the file and print out the headers accordingly 
      l = modelname
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      
      outcome = split_trained_line[2]
      outcome = str(outcome)

      no_columns = len(predictorsStr)

      return render_template("predictmodel.html", headers = predictorsStr, no_columns = no_columns, outcome = outcome, modelname = modelname)
    except:
      flash('Error: There are no models trained yet. Please train a model first')
      return redirect('/selectcols')

  flash("Please login to view this page")
  return redirect("/")

# Method to load in the inputted values and predict
@app.route('/modelrunanalysis', methods=['GET', 'POST'])
def modelrunanalysis():
  if 'username' in session:
    username = session['username']

    # person inputs data then the persisted model is used to predict
    if request.method == 'POST':
      
      try:
        values = request.form 

        # do prediction based on inputted values
        
        # load in the inputs from the form
        modeln = values['modelName']
        modeln = modeln.strip()

        # read pickled model 
        full_path_to_downloaded_model = '/app/files/' + modeln + '.pkl'
        model = joblib.load(full_path_to_downloaded_model)

        l = modeln
        f = open('/app/files/trainedModels.txt', 'r')
        lines = f.readlines()
        for line in lines:
          if l in line:
            trained_line = line
        f.close()
        # load inputs from file 
        split_trained_line = trained_line.split("_")
        predictors = split_trained_line[1]
        predictors = predictors[1:-1]
        predict_strip = [ x.strip() for x in predictors.split(',') ]
        predict_strip2 = [ x[1:-1] for x in predict_strip ]
        predict_array = np.asarray(predict_strip2)

        false_neg_rate = split_trained_line[3]
        true_pos_rate = split_trained_line[4]
        
        loaded_values = []
        no_columns = len(predict_array)

        for x in range(no_columns):
          string = predict_array[x]
          string = str(string)  
          loaded_values.append(values[string])

        # processing 
        processed_values = process_one_line(loaded_values)
        
        # predict
        
        # check values have been inputted
        try:
          processed_values2 = [ float(x) for x in processed_values ]
          pred_input = np.asarray(processed_values2)
          pred_input_reshape = pred_input.reshape(1, -1)
          prediction2 = model.predict(pred_input_reshape)

          # make predictions on the up-sampled data
          prob_prediction = model.predict_proba(pred_input_reshape)
          class1_prediction = prob_prediction[:,1]
          class0_prediction = prob_prediction[:,0]
          
          if (str(class1_prediction) == '[0.]'):
            final_pred_class1 = [0]
            final_pred_class0 = [1]
          
          else: 
            # Get mappings
            # Concept from http://blog.data-miners.com/2009/09/adjusting-for-oversampling.html
            value0 = split_trained_line[12]
            value1 = split_trained_line[13].strip()
            value0 = float(value0)
            value1 = float(value1)
            class0, class1 = calc_prob(value0, value1)

            # map the prediction made on the up-sampled data to the original
            class1_prediction_mapped = class1_prediction / class1
            class0_prediction_mapped = class0_prediction / class0

            final_pred_class1 = class1_prediction_mapped / (class1_prediction_mapped + class0_prediction_mapped)
            final_pred_class1 = np.round(final_pred_class1, 5)
            final_pred_class1 = final_pred_class1*100
            final_pred_class0 = class0_prediction_mapped / (class1_prediction_mapped + class0_prediction_mapped)
            final_pred_class0 = np.round(final_pred_class0, 5)

        except:
          flash('Error: None of the predictor fields can be empty. Please try again')
          return redirect('/choosemodel')

        final_pred = "This patient has a "
        final_pred += str(final_pred_class1)[1:-1]
        final_pred += "%"
        final_pred += " chance of having this outcome. Since the average risk for this complication is "
        final_pred += str(split_trained_line[14])
        final_pred += "%"
        final_pred += " any risk above "
        final_pred += str(split_trained_line[15].strip())
        final_pred += "%"
        final_pred += " is considered to be HIGH risk compared to the average population."

      
        # display other metrics on the model
        tp = split_trained_line[3]
        fp = split_trained_line[4]
        tn = split_trained_line[5]
        fn = split_trained_line[6]
        roc = split_trained_line[7]
        precision1 = split_trained_line[8]
        precision2 = split_trained_line[9]
        recall1 = split_trained_line[10]
        recall2 = split_trained_line[11]

        return render_template('predictionsuccessful.html', final_pred = final_pred, tp = tp, fp=fp, tn=tn, fn=fn, roc=roc, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2)

      except:
        flash('Error: There are no models trained yet. Please train a model first')
        return redirect('/selectcols')       
  
  flash("Please login to view this page")
  return redirect("/")

# Admin predict common complications at once
@app.route('/adminpredictcomp')
def adminpredictcomp():
  if 'username' in session:
    try:
      username = session['username']
      
      l = 'ren30'
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          topline = line
      f.close()
      
      # f = open('/app/files/trainedModels.txt', 'r')
      # topline = f.readline()
      # f.close()

      topline = str(topline)
      split_topline = topline.split("_")
      comorbidities = split_topline[1]
      comorbidities = comorbidities[1:-1]
      comorbidities_strip = [ x.strip() for x in comorbidities.split(',') ]
      comorbidities_str = [ x[1:-1] for x in comorbidities_strip ]
      comorbidities_array = np.array(comorbidities_str)

      age = comorbidities_array[0]

      # remove age from array
      comorbidities_array_noage = []
      for c in comorbidities_array:
        if c != 'age':
          comorbidities_array_noage.append(c)
  
      comorbidities_array_noage_len = len(comorbidities_array_noage)

      return render_template("adminpredictcomp.html", comorbid = comorbidities_array_noage, length = comorbidities_array_noage_len)
    except:
      flash('Sorry, no models have been created yet, so no predictions can be made.')
      flash('Please build a model first.')
      return redirect("/selectcols")
  
  flash("Please login to view this page")
  return redirect("/")

# Method to predict general complications 
@app.route('/adminpredictcompafter', methods=['GET', 'POST'])
def adminpredictcompafter():
  if 'username' in session:
    username = session['username']

    try:
      
      ren = 'ren30'
      dvt = 'dvt60'
      pe = 'pe60'
      dhosp = 'dhosp90'
      dout90 = 'dout90'
      pne30 = 'pne30'
      mi30 = 'mi30'
      readm = 'readm'

      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if ren in line:
          ren_line = line
        elif dvt in line:
          dvt_line = line
        elif pe in line:
          pe_line = line
        elif dhosp in line:
          dhosp_line = line
        elif dout90 in line:
          dout90_line = line
        elif pne30 in line:
          pne30_line = line
        elif mi30 in line:
          mi30_line = line
        elif readm in line:
          readm_line = line
      f.close()
      
      # f = open('/app/files/trainedModels.txt', 'r')
      # topline = f.readline()
      # f.close()

      ren_line = str(ren_line)
      split_topline = ren_line.split("_")
      comorbidities = split_topline[1]
      comorbidities = comorbidities[1:-1]
      comorbidities_strip = [ x.strip() for x in comorbidities.split(',') ]
      comorbidities_str = [ x[1:-1] for x in comorbidities_strip ]
      comorbidities_array = np.array(comorbidities_str)

      age = comorbidities_array[0]

      # remove age from array
      comorbidities_array_noage = []
      for c in comorbidities_array:
        if c != 'age':
          comorbidities_array_noage.append(c)
      
      comorbidities_array_noage_len = len(comorbidities_array_noage)

      ticks = request.form.getlist('co_morbid')
      age = request.form['age']

      if not age:
        flash('You left the age empty, please try again')
        return redirect('/adminpredictcomp')

      age = int(age)

      values_in = comorbidities_array[:]
      values_in = np.array(values_in)
      
      for v in range(len(values_in)):
        #if values_in[v] == 'age':
          #values_in[v] = age
        #else:
        for t in range(len(ticks)):
          if ticks[t] == values_in[v]:
            values_in[v] = 1
      
      #values_in = [ int(x) for x in values_in ]

      for v in range(len(values_in)):
        if values_in[v] == 'age':
          values_in[v] = age
        elif values_in[v] != '1':
          values_in[v] = '0'

      values_in = [ int(x) for x in values_in ]

      # load in models 
      ren30_model_path = 'files/' + 'ren30' + '.pkl'
      dvt60_model_path = '/app/files/' + 'dvt60' + '.pkl'
      pe60_model_path = '/app/files/' + 'pe60' + '.pkl'
      pne30_model_path = '/app/files/' + 'pne30' + '.pkl'
      dhosp90_model_path = '/app/files/' + 'dhosp90' + '.pkl'
      dout90_model_path = '/app/files/' + 'dout90' + '.pkl'
      mi30_model_path = '/app/files/' + 'mi30' + '.pkl'
      readm_model_path = '/app/files/' + 'readm' + '.pkl'
      
      ren30model = joblib.load(ren30_model_path)
      dvt60model = joblib.load(dvt60_model_path)
      pe60model = joblib.load(pe60_model_path)
      pne30model = joblib.load(pne30_model_path)
      dhosp90model = joblib.load(dhosp90_model_path)
      dout90model = joblib.load(dout90_model_path)
      mi30model = joblib.load(mi30_model_path)
      readmmodel = joblib.load(readm_model_path)

      models = [ren30model, dvt60model, pe60model, pne30model, dhosp90model, dout90model, mi30model, readmmodel]

      models_string = ['ren30', 'dvt60', 'pe60', 'pne30', 'dhosp90', 'dout90', 'mi30', 'readm']
      models_string_len = len(models_string)

      # predict outcomes
      processed_values2 = [ float(x) for x in values_in ]
      pred_input = np.asarray(processed_values2)
      pred_input_reshape = pred_input.reshape(1, -1)

      # load in the rates to display with the prediction
      
      # ren_false_neg_rate = split_topline[3]
      # ren_pos_rate = split_topline[4]

      # create arrays for the metrics 
      tp_array = []
      fp_array = []
      tn_array = []
      fn_array = []
      roc_array = []
      prec1_array = []
      prec2_array = []
      rec1_array = []
      rec2_array = []

      ren_tp, ren_fp, ren_tn, ren_fn, ren_roc, ren_precision1, ren_precision2, ren_recall1, ren_recall2, ren_value0, ren_value1 = get_rates_values(ren_line)
      dvt_tp, dvt_fp, dvt_tn, dvt_fn, dvt_roc, dvt_precision1, dvt_precision2, dvt_recall1, dvt_recall2, dvt_value0, dvt_value1 = get_rates_values(dvt_line)
      pe_tp, pe_fp, pe_tn, pe_fn, pe_roc, pe_precision1, pe_precision2, pe_recall1, pe_recall2, pe_value0, pe_value1 = get_rates_values(pe_line)
      pne_tp, pne_fp, pne_tn, pne_fn, pne_roc, pne_precision1, pne_precision2, pne_recall1, pne_recall2, pne_value0, pne_value1 = get_rates_values(pne30_line)
      dhosp_tp, dhosp_fp, dhosp_tn, dhosp_fn, dhosp_roc, dhosp_precision1, dhosp_precision2, dhosp_recall1, dhosp_recall2, dhosp_value0, dhosp_value1 = get_rates_values(dhosp_line)
      dout_tp, dout_fp, dout_tn, dout_fn, dout_roc, dout_precision1, dout_precision2, dout_recall1, dout_recall2, dout_value0, dout_value1 = get_rates_values(dout90_line)
      mi_tp, mi_fp, mi_tn, mi_fn, mi_roc, mi_precision1, mi_precision2, mi_recall1, mi_recall2, mi_value0, mi_value1 = get_rates_values(mi30_line)
      readm_tp, readm_fp, readm_tn, readm_fn, readm_roc, readm_precision1, readm_precision2, readm_recall1, readm_recall2, readm_value0, readm_value1 = get_rates_values(readm_line)
      
      # store all the metrics into the arrays 
      tp_array.extend((ren_tp, dvt_tp, pe_tp, pne_tp, dhosp_tp, dout_tp, mi_tp, readm_tp))
      fp_array.extend((ren_fp, dvt_fp, pe_fp, pne_fp, dhosp_fp, dout_fp, mi_fp, readm_fp))
      tn_array.extend((ren_tn, dvt_tn, pe_tn, pne_tn, dhosp_tn, dout_tn, mi_tn, readm_tn))
      fn_array.extend((ren_fn, dvt_fn, pe_fn, pne_fn, dhosp_fn, dout_fn, mi_fn, readm_fn))
      roc_array.extend((ren_roc, dvt_roc, pe_roc, pne_roc, dhosp_roc, dout_roc, mi_roc, readm_roc))
      prec1_array.extend((ren_precision1, dvt_precision1, pe_precision1, pne_precision1, dhosp_precision1, dout_precision1, mi_precision1, readm_precision1))
      prec2_array.extend((ren_precision2, dvt_precision2, pe_precision2, pne_precision2, dhosp_precision2, dout_precision2, mi_precision2, readm_precision2))
      rec1_array.extend((ren_recall1, dvt_recall1, pe_recall1, pne_recall1, dhosp_recall1, dout_recall1, mi_recall1, readm_recall1))
      rec2_array.extend((ren_recall2, dvt_recall2, pe_recall2, pne_recall2, dhosp_recall2, dout_recall2, mi_recall2, readm_recall2))

      # create array to store final predictions in 

      predictions = []

      ren_prediction = predict_model(ren30model, pred_input_reshape, ren_value0, ren_value1, ren_line)
      dvt_prediction = predict_model(dvt60model, pred_input_reshape, dvt_value0, dvt_value1, dvt_line)
      pe_prediction = predict_model(pe60model, pred_input_reshape, pe_value0, pe_value1, pe_line)
      pne_prediction = predict_model(pne30model, pred_input_reshape, pne_value0, pne_value1, pne30_line)
      dhosp_prediction = predict_model(dhosp90model, pred_input_reshape, dhosp_value0, dhosp_value1, dhosp_line)
      dout_prediction = predict_model(dout90model, pred_input_reshape, dout_value0, dout_value1, dout90_line)
      mi_prediction = predict_model(mi30model, pred_input_reshape, mi_value0, mi_value1, mi30_line)
      readm_prediction = predict_model(readmmodel, pred_input_reshape, readm_value0, readm_value1, readm_line)
      
      # create array to store final predictions in 

      predictions = []

      predictions.append(ren_prediction)
      predictions.append(dvt_prediction)
      predictions.append(pe_prediction)
      predictions.append(pne_prediction)
      predictions.append(dhosp_prediction)
      predictions.append(dout_prediction)
      predictions.append(mi_prediction)
      predictions.append(readm_prediction)

      # return str(predictions)
      return render_template("adminpredictcompafter.html", comorbid = comorbidities_array_noage, length = comorbidities_array_noage_len, models_string = models_string, models_string_len = models_string_len, predictions = predictions, tp_array= tp_array, fp_array=fp_array, tn_array=tn_array, fn_array=fn_array, roc_array=roc_array, prec1_array=prec1_array, prec2_array=prec2_array, rec1_array=rec1_array, rec2_array=rec2_array)
      
    except:
      flash('Error: There is no data or no models have been trained yet. Please train models first.')
      return redirect('/selectcols')
  flash("Please login to view this page")
  return redirect("/")

# ==================================================================
# Surgeon methods
# The following methods are for the user interface
# Includes:
# Method to predict general complications 
# (renal, dvt, pe, dhosp90, dout90, pne) 
# Method to choose another model from admin generated model 
# and predict using this model
# ==================================================================

# Method to predict general complications 
@app.route('/userpredictcomplications', methods=['GET', 'POST'])
def userpredictcomplications():
  if 'username' in session:
    username = session['username']

    try:
      
      ren = 'ren30'
      dvt = 'dvt60'
      pe = 'pe60'
      dhosp = 'dhosp90'
      dout90 = 'dout90'
      pne30 = 'pne30'
      mi30 = 'mi30'
      readm = 'readm'

      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if ren in line:
          ren_line = line
        elif dvt in line:
          dvt_line = line
        elif pe in line:
          pe_line = line
        elif dhosp in line:
          dhosp_line = line
        elif dout90 in line:
          dout90_line = line
        elif pne30 in line:
          pne30_line = line
        elif mi30 in line:
          mi30_line = line
        elif readm in line:
          readm_line = line
      f.close()

      ren_line = str(ren_line)
      split_topline = ren_line.split("_")
      comorbidities = split_topline[1]
      comorbidities = comorbidities[1:-1]
      comorbidities_strip = [ x.strip() for x in comorbidities.split(',') ]
      comorbidities_str = [ x[1:-1] for x in comorbidities_strip ]
      comorbidities_array = np.array(comorbidities_str)

      age = comorbidities_array[0]

      # remove age from array
      comorbidities_array_noage = []
      for c in comorbidities_array:
        if c != 'age':
          comorbidities_array_noage.append(c)
      
      comorbidities_array_noage_len = len(comorbidities_array_noage)

      ticks = request.form.getlist('co_morbid')
      age = request.form['age']

      if not age:
        flash('You left the age empty, please try again')
        return redirect('/userhomepage')

      age = int(age)

      values_in = comorbidities_array[:]
      values_in = np.array(values_in)
      
      for v in range(len(values_in)):
        #if values_in[v] == 'age':
          #values_in[v] = age
        #else:
        for t in range(len(ticks)):
          if ticks[t] == values_in[v]:
            values_in[v] = 1
      
      #values_in = [ int(x) for x in values_in ]

      for v in range(len(values_in)):
        if values_in[v] == 'age':
          values_in[v] = age
        elif values_in[v] != '1':
          values_in[v] = '0'

      values_in = [ int(x) for x in values_in ]

      # load in models 
      ren30_model_path = 'files/' + 'ren30' + '.pkl'
      dvt60_model_path = '/app/files/' + 'dvt60' + '.pkl'
      pe60_model_path = '/app/files/' + 'pe60' + '.pkl'
      pne30_model_path = '/app/files/' + 'pne30' + '.pkl'
      dhosp90_model_path = '/app/files/' + 'dhosp90' + '.pkl'
      dout90_model_path = '/app/files/' + 'dout90' + '.pkl'
      mi30_model_path = '/app/files/' + 'mi30' + '.pkl'
      readm_model_path = '/app/files/' + 'readm' + '.pkl'
      
      ren30model = joblib.load(ren30_model_path)
      dvt60model = joblib.load(dvt60_model_path)
      pe60model = joblib.load(pe60_model_path)
      pne30model = joblib.load(pne30_model_path)
      dhosp90model = joblib.load(dhosp90_model_path)
      dout90model = joblib.load(dout90_model_path)
      mi30model = joblib.load(mi30_model_path)
      readmmodel = joblib.load(readm_model_path)

      models = [ren30model, dvt60model, pe60model, pne30model, dhosp90model, dout90model, mi30model, readmmodel]

      models_string = ['ren30', 'dvt60', 'pe60', 'pne30', 'dhosp90', 'dout90', 'mi30', 'readm']
      
      models_string_len = len(models_string)

      # predict outcomes
      #processed_values2 = map(float,values_in)
      processed_values2 = [ float(x) for x in values_in ]
      pred_input = np.asarray(processed_values2)
      pred_input_reshape = pred_input.reshape(1, -1)

      # load in the rates to display with the prediction
      
      # ren_false_neg_rate = split_topline[3]
      # ren_pos_rate = split_topline[4]
      # ren_value0 = float(split_topline[5])
      # ren_value1 = float(split_topline[6].strip())

      # create arrays for the metrics 
      tp_array = []
      fp_array = []
      tn_array = []
      fn_array = []
      roc_array = []
      prec1_array = []
      prec2_array = []
      rec1_array = []
      rec2_array = []

      ren_tp, ren_fp, ren_tn, ren_fn, ren_roc, ren_precision1, ren_precision2, ren_recall1, ren_recall2, ren_value0, ren_value1 = get_rates_values(ren_line)
      dvt_tp, dvt_fp, dvt_tn, dvt_fn, dvt_roc, dvt_precision1, dvt_precision2, dvt_recall1, dvt_recall2, dvt_value0, dvt_value1 = get_rates_values(dvt_line)
      pe_tp, pe_fp, pe_tn, pe_fn, pe_roc, pe_precision1, pe_precision2, pe_recall1, pe_recall2, pe_value0, pe_value1 = get_rates_values(pe_line)
      pne_tp, pne_fp, pne_tn, pne_fn, pne_roc, pne_precision1, pne_precision2, pne_recall1, pne_recall2, pne_value0, pne_value1 = get_rates_values(pne30_line)
      dhosp_tp, dhosp_fp, dhosp_tn, dhosp_fn, dhosp_roc, dhosp_precision1, dhosp_precision2, dhosp_recall1, dhosp_recall2, dhosp_value0, dhosp_value1 = get_rates_values(dhosp_line)
      dout_tp, dout_fp, dout_tn, dout_fn, dout_roc, dout_precision1, dout_precision2, dout_recall1, dout_recall2, dout_value0, dout_value1 = get_rates_values(dout90_line)
      mi_tp, mi_fp, mi_tn, mi_fn, mi_roc, mi_precision1, mi_precision2, mi_recall1, mi_recall2, mi_value0, mi_value1 = get_rates_values(mi30_line)
      readm_tp, readm_fp, readm_tn, readm_fn, readm_roc, readm_precision1, readm_precision2, readm_recall1, readm_recall2, readm_value0, readm_value1 = get_rates_values(readm_line)
      
      # store all the metrics into the arrays 
      tp_array.extend((ren_tp, dvt_tp, pe_tp, pne_tp, dhosp_tp, dout_tp, mi_tp, readm_tp))
      fp_array.extend((ren_fp, dvt_fp, pe_fp, pne_fp, dhosp_fp, dout_fp, mi_fp, readm_fp))
      tn_array.extend((ren_tn, dvt_tn, pe_tn, pne_tn, dhosp_tn, dout_tn, mi_tn, readm_tn))
      fn_array.extend((ren_fn, dvt_fn, pe_fn, pne_fn, dhosp_fn, dout_fn, mi_fn, readm_fn))
      roc_array.extend((ren_roc, dvt_roc, pe_roc, pne_roc, dhosp_roc, dout_roc, mi_roc, readm_roc))
      prec1_array.extend((ren_precision1, dvt_precision1, pe_precision1, pne_precision1, dhosp_precision1, dout_precision1, mi_precision1, readm_precision1))
      prec2_array.extend((ren_precision2, dvt_precision2, pe_precision2, pne_precision2, dhosp_precision2, dout_precision2, mi_precision2, readm_precision2))
      rec1_array.extend((ren_recall1, dvt_recall1, pe_recall1, pne_recall1, dhosp_recall1, dout_recall1, mi_recall1, readm_recall1))
      rec2_array.extend((ren_recall2, dvt_recall2, pe_recall2, pne_recall2, dhosp_recall2, dout_recall2, mi_recall2, readm_recall2))

      # create array to store final predictions in 

      predictions = []

      ren_prediction = predict_model(ren30model, pred_input_reshape, ren_value0, ren_value1, ren_line)
      dvt_prediction = predict_model(dvt60model, pred_input_reshape, dvt_value0, dvt_value1, dvt_line)
      pe_prediction = predict_model(pe60model, pred_input_reshape, pe_value0, pe_value1, pe_line)
      pne_prediction = predict_model(pne30model, pred_input_reshape, pne_value0, pne_value1, pne30_line)
      dhosp_prediction = predict_model(dhosp90model, pred_input_reshape, dhosp_value0, dhosp_value1, dhosp_line)
      dout_prediction = predict_model(dout90model, pred_input_reshape, dout_value0, dout_value1, dout90_line)
      mi_prediction = predict_model(mi30model, pred_input_reshape, mi_value0, mi_value1, mi30_line)
      readm_prediction = predict_model(readmmodel, pred_input_reshape, readm_value0, readm_value1, readm_line)

      predictions.append(ren_prediction)
      predictions.append(dvt_prediction)
      predictions.append(pe_prediction)
      predictions.append(pne_prediction)
      predictions.append(dhosp_prediction)
      predictions.append(dout_prediction)
      predictions.append(mi_prediction)
      predictions.append(readm_prediction)

      # return ''' hellos ''' + str(readm_prediction)  + ''' '''
      return render_template("userhomepageafterpredict.html", username=username, comorbid = comorbidities_array_noage, length = comorbidities_array_noage_len, models_string = models_string, models_string_len = models_string_len, predictions = predictions, tp_array= tp_array, fp_array=fp_array, tn_array=tn_array, fn_array=fn_array, roc_array=roc_array, prec1_array=prec1_array, prec2_array=prec2_array, rec1_array=rec1_array, rec2_array=rec2_array)
      
      #return render_template("userhomepageafterpredict.html", comorbid = comorbidities_array_noage, length = comorbidities_array_noage_len, models_string = models_string, models_string_len = models_string_len, predictions = predictions, username=username)
    
    except:
      flash('Error: There is no data or no models have been trained yet. Please contact your system admin to train models first.')
      return redirect('/logout')

  flash("Please login to view this page")
  return redirect("/")

# Method to generate page to choose an admin generated model
@app.route('/userchoosemodel', methods=['GET', 'POST'])
def userchoosemodel():
  if 'username' in session:
    username = session['username']
    try: 

    # file1 = open('/app/files/trainedmodels.txt', 'r')
    
      path = "trainedModels.txt"
      file1 = open(os.path.join(app.config['UPLOAD_FOLDER'], path), 'r')
      
      modelnames = file1.readlines()
      
      names = []
      for model in modelnames:
        model_array = model.split("_")
        names.append(model_array[0])

      file1.close()

      
      return render_template("userchoosemodel.html", modelnames = names)
    except:
      flash('Error: There are no models trained yet. ')
      flash('Please contact your system admin to train models first.')
      return redirect('/logout')

  flash("Please login to view this page")
  return redirect("/")

# Method to generate page which allows input of values used to predict an outcome
@app.route('/userpredictmodel', methods=['GET', 'POST'])
def userpredictmodel():
  if 'username' in session:
    username = session['username']

    try:
      values = request.form 
      modelname = values['modelnames']
      modelname = str(modelname)

      if not (modelname):
        flash("Modelname cannot be empty.")
        return redirect(url_for('userchoosemodel'))

      # read the file and print out the headers accordingly 
      l = modelname
      f = open('/app/files/trainedModels.txt', 'r')
      lines = f.readlines()
      for line in lines:
        if l in line:
          trained_line = line
      f.close()

      trained_line = str(trained_line)
      split_trained_line = trained_line.split("_")
      predictors = split_trained_line[1]
      predictors = predictors[1:-1]
      predictors_strip = [ x.strip() for x in predictors.split(',') ]
      predictorsStr = [ x[1:-1] for x in predictors_strip ]
      
      outcome = split_trained_line[2]
      outcome = str(outcome)

      no_columns = len(predictorsStr)

      return render_template("userpredictmodel.html", headers = predictorsStr, no_columns = no_columns, outcome = outcome, modelname = modelname)
    except:
      flash('Error: There is no data or no models have been trained yet. ')
      flash('Please contact your system admin to train models first.')
      return redirect('/logout')

  flash("Please login to view this page")
  return redirect("/")

# Method to use inputted values to predict an outcome
@app.route('/usermodelrunanalysis', methods=['GET', 'POST'])
def usermodelrunanalysis():
  if 'username' in session:
    username = session['username']

    # person inputs data then the persisted model is used to predict
    if request.method == 'POST':
      
      try:
        values = request.form 

        # do prediction based on inputted values
        
        # load in the inputs from the form

        modeln = values['modelName']
        modeln = modeln.strip()

        full_path_to_downloaded_model = '/app/files/' + modeln + '.pkl'
        model = joblib.load(full_path_to_downloaded_model)

        l = modeln
        f = open('/app/files/trainedModels.txt', 'r')
        lines = f.readlines()
        for line in lines:
          if l in line:
            trained_line = line
        f.close()

        split_trained_line = trained_line.split("_")
        predictors = split_trained_line[1]
        predictors = predictors[1:-1]
        predict_strip = [ x.strip() for x in predictors.split(',') ]
        predict_strip2 = [ x[1:-1] for x in predict_strip ]
        predict_array = np.asarray(predict_strip2)

        false_neg_rate = split_trained_line[3]
        true_pos_rate = split_trained_line[4]
        
        loaded_values = []
        no_columns = len(predict_array)

        # check values have been inputted
        for x in range(no_columns):
          string = predict_array[x]
          string = str(string)  
          loaded_values.append(values[string])        

        # processing 
        processed_values = process_one_line(loaded_values)

        # check values have been inputted
        try:
          processed_values2 = [ float(x) for x in processed_values ]
          pred_input = np.asarray(processed_values2)
          pred_input_reshape = pred_input.reshape(1, -1)
          prediction2 = model.predict(pred_input_reshape)

          # make predictions on the up-sampled data
          prob_prediction = model.predict_proba(pred_input_reshape)
          class1_prediction = prob_prediction[:,1]
          class0_prediction = prob_prediction[:,0]
          
          if (str(class1_prediction) == '[0.]'):
            final_pred_class1 = [0]
            final_pred_class0 = [1]
          
          else: 
            # Get mappings
            # Concept from http://blog.data-miners.com/2009/09/adjusting-for-oversampling.html
            value0 = split_trained_line[12]
            value1 = split_trained_line[13].strip()
            value0 = float(value0)
            value1 = float(value1)
            class0, class1 = calc_prob(value0, value1)

            # map the prediction made on the up-sampled data to the original
            class1_prediction_mapped = class1_prediction / class1
            class0_prediction_mapped = class0_prediction / class0

            final_pred_class1 = class1_prediction_mapped / (class1_prediction_mapped + class0_prediction_mapped)
            final_pred_class1 = np.round(final_pred_class1, 5)
            final_pred_class1 = final_pred_class1*100
            final_pred_class0 = class0_prediction_mapped / (class1_prediction_mapped + class0_prediction_mapped)
            final_pred_class0 = np.round(final_pred_class0, 5)

        except:
          flash('Error: None of the predictor fields can be empty. Please try again')
          return redirect('/choosemodel')

        final_pred = "This patient has a "
        final_pred += str(final_pred_class1)[1:-1]
        final_pred += "%"
        final_pred += " chance of having this outcome. Since the average risk for this complication is "
        final_pred += str(split_trained_line[14])
        final_pred += "%"
        final_pred += " any risk above "
        final_pred += str(split_trained_line[15].strip())
        final_pred += "%"
        final_pred += " is considered to be HIGH risk compared to the average population."
        
        # display other metrics on the model
        tp = split_trained_line[3]
        fp = split_trained_line[4]
        tn = split_trained_line[5]
        fn = split_trained_line[6]
        roc = split_trained_line[7]
        precision1 = split_trained_line[8]
        precision2 = split_trained_line[9]
        recall1 = split_trained_line[10]
        recall2 = split_trained_line[11]

        return render_template('userpredictionsuccessful.html', final_pred = final_pred, tp = tp, fp=fp, tn=tn, fn=fn, roc=roc, precision1=precision1, precision2=precision2, recall1=recall1, recall2=recall2)

      except:
        flash('Error: There are no models trained yet. Please contact your system admin to train models first.')
        return redirect('/logout')

  flash("Please login to view this page")
  return redirect("/")

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)
