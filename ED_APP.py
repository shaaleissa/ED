# import lib
from flask import Flask, jsonify, render_template, url_for, redirect, request, session, flash, logging, abort
from functools import wraps
import pandas
import sklearn
import pickle
import os
import joblib 
from flask_mail import Mail, Message
from passlib.hash import sha256_crypt
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import random, string
from random import choice
from datetime import datetime
import gc
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import *

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import *
import xgboost as xgb
# import lightgbm as lgbm

from wtforms import Form, StringField, validators
from wtforms.fields import DateField
from sklearn import neural_network, ensemble
from sklearn import metrics
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from wtforms import Form, StringField, TextAreaField, PasswordField, validators, IntegerField
from wtforms import BooleanField, DateTimeField, SelectField
from wtforms.validators import InputRequired, DataRequired, Email, Length, ValidationError
import pymongo
from flask_pymongo import PyMongo
from pymongo import MongoClient

# save vir env to app
app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True)

# config the email
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'graduation.project.ai.123@gmail.com'
app.config['MAIL_PASSWORD'] = 'tvviphokfuafgprr'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

# save the data & model into folders & config them
app.secret_key = '4zJW=nyT[Bk:4uuY'
UPLOAD_FOLDER = './static/data'  # .csv
UPLOADED_ITEMS_DEST = './static/model'  # .sav
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOADED_ITEMS_DEST'] = UPLOADED_ITEMS_DEST


def connection():  # establish the database connection and return the database and client

    app.mongodb_client = MongoClient("mongodb+srv://graduationpojectai:W18UGYcRxlaBZY1X@cluster0.sbpeohk.mongodb.net/")
    app.database = app.mongodb_client["GPDB"]

    return app.database, app.mongodb_client


def email(subject, recipient, html):  # gets the subject ,recipient and html body of email and return the email message to be sent
    return Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[recipient], html=html)

class PMRN(Form):
    patientID = IntegerField('Patient MRN', [
        validators.NumberRange(min=1000000, max=999999999, message="Patient MRN must be 7 to 9 digits")])

@app.route("/")
def index():
    return render_template('index.html')

def logingout():
    if 'logged_in' in session:
        session.pop('logged_in')
        session.pop('role')
        session.pop('username')

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return render_template('404.html')

    return wrap

@app.route("/logout")
@login_required
def logout():
    logingout()
    return render_template('index.html')


# DON'T REMOVE THIS
mrn = None
@app.route("/diagnosishistory", methods=['GET', 'POST'])
@login_required
def diagnosishistory():
    usertype = session['role']
    if usertype == 'medical specialist':
        global mrn
        form = PMRN(request.form)

        if request.method == 'POST' and form.validate():
            mrn = form.patientID.data
            flash('Patient does not exist.', 'error')
            return redirect(url_for('diagnosishistory'))

        elif mrn is not None:
            try:
                db, client = connection()

                final_ls = []
                if mrn is not None:
                    pipeline = [{"$unwind": "$predicted_disease"},
                                {"$project": {"_id": 1, "test_id": 1, "disease": "$predicted_disease.disease",
                                              "test_date": 1,
                                              "prediction": "$predicted_disease.prediction",
                                              "accuracy": "$predicted_disease.accuracy"}}]

                    results = db.test.aggregate([{"$match": {"mrn": mrn}}, *pipeline])

                    for r in results:
                        final_ls.append(r)

            finally:
                client.close()
            mrn = None
            return render_template('MSdiagnoseHistory.html', result=final_ls, form=form, name='',
                                   usertype=usertype)
        else:
            print ('odddddddd')
            return render_template('MSdiagnoseHistory.html', form=form, name='', usertype=usertype)

    elif usertype == 'registered user':
        RUID = session['username']

        try:
            db, client = connection()
            mrn = db.patient.find_one({'username': RUID}, {'_id': 1})
            final_ls = []
            if mrn is not None:
                pipeline = [{"$unwind": "$predicted_disease"},
                            {"$project": {"_id": 1, "test_id": 1, "disease": "$predicted_disease.disease",
                                          "test_date": 1,
                                          "prediction": "$predicted_disease.prediction",
                                          "accuracy": "$predicted_disease.accuracy"}}]

                results = db.test.aggregate([{"$match": {"mrn": mrn['_id']}}, *pipeline])

                for r in results:
                    final_ls.append(r)

        finally:
            client.close()
        return render_template('MSdiagnoseHistory.html', result=final_ls, name='', usertype=usertype)
    else:
        return render_template('404.html')


# Rebuild Model
@app.route('/rebuildModel')
@login_required
def display():
    if session['role'] == 'admin':
        db, client = connection()

        # fetch the temp models
        tempModels_cursor = db.tempmodel.find({}, {"_id": 0})

        tempModels_list = list(tempModels_cursor)
        tempModels_count = len(tempModels_list)

        # fetch the active models
        activeModels_cursor = db.model.find({"Active": 1}, {"_id": 0})

        activeModels_list = list(activeModels_cursor)
        activeModels_count = len(activeModels_list)

        if 'updateM' in session:
            flash("Success! Diagnostic Model has been Updated", 'info')
            session.pop('updateM')
        return render_template('adminRebuildModel.html', tempModels=tempModels_list, tempModels_count=tempModels_count,
                               activeModels=activeModels_list, activeModels_count=activeModels_count)
    else:
        return render_template('404.html')

@app.route('/generateModel', methods=['POST'])
def generateModel():
    # Start first if  (General, no need to add more)
    if request.method == 'POST':
        training = request.form['trainingP']
        test = (100 - int(training)) / 100
        disease = request.form.get('diseaseList')
        file = request.files['dataFile']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        savedFile = './static/data/' + filename

        # .................................................
        # ...........ADD THE DISEASE MODEL HERE ..........
        # .................................................

        # Start second if (Which has the best technique for each disease, add the new techniques )

        if (disease == 'Diabetes'):
            names = [1, 2, 3, 4]  # All Attributes
            items1 = [1, 2, 3]  # Attributes without class
            items = [4]  # Class Attribute

            data = pandas.read_csv(savedFile, names=names)  # read .csv file

            train, test = train_test_split(data, test_size=test, random_state=0)  # divid the data

            x_data, x_class = train.filter(items1), train.filter(items)  # save data for training

            y_data, y_class = test.filter(items1), test.filter(items)  # save data for testing

            # define the technique with the best parameters

            clf = sklearn.neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5), learning_rate='constant',
                                                       learning_rate_init=0.3, random_state=0)

        elif (disease == 'Chronic Kidney Disease'):
            names = [1, 2, 3]
            items1 = [1, 2]
            items = [3]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = sklearn.neural_network.MLPClassifier(solver='lbfgs', activation='identity', hidden_layer_sizes=(100),
                                                       random_state=0)

        elif (disease == 'Coronary Heart Disease'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            items1 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            items = [1]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = NuSVC(kernel='poly', nu=0.3, random_state=42)

        elif (disease == 'Rheumatoid Arthritis Disease'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
            items = [27]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = sklearn.svm.SVC(kernel='rbf', C=25, gamma=0.001)

        elif (disease == 'Asthma Disease'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            items = [10]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = sklearn.svm.SVC(C=10, gamma=0.0001, kernel='rbf')

        elif (disease == 'Thyroid Cancer'):
            names = [1, 2, 3, 4, 5, 6, 7, 8]
            items1 = [1, 2, 3, 4, 5, 6, 7]
            items = [8]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = sklearn.neural_network.MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(100),
                                                       learning_rate='constant', alpha=0.01)

        elif (disease == 'Hypothyroidism'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            items1 = [2, 4, 7, 9, 10]
            items = [19]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            estimators = []
            model1 = KNeighborsClassifier()
            estimators.append(('knn1', model1))
            model2 = sklearn.svm.SVC(C=9, gamma=1, kernel='linear', probability=True)
            estimators.append(('svm1', model2))
            model3 = sklearn.svm.SVC(C=9, gamma=1, kernel='rbf', probability=True)
            estimators.append(('svm2', model3))
            clf = sklearn.ensemble.VotingClassifier(estimators, voting='soft')

        elif (disease == 'Prostate Cancer'):
            names = [1, 2, 3, 4, 5, 6, 7, 9, 10]
            items1 = [3, 4, 5, 6]
            items = [10]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            x_class = x_class.values.ravel()
            clf = sklearn.svm.SVC(C=100, gamma=0.1)

        elif (disease == 'Alzheimer’s Disease'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            items = [14]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = SVC(C=4, gamma=1, random_state=0)

        elif (disease == 'Glaucoma'):
            names = [1, 2, 3, 4, 5, 6, 7, 8]
            items1 = [1, 2, 3, 4, 5, 6, 7]
            items = [8]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = KNeighborsClassifier(metric='manhattan', n_neighbors=17)

        elif (disease == 'Lung Cancer'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            items = [16]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = sklearn.svm.SVC(C=15, gamma=0.001, kernel='rbf')

        elif (disease == 'ADHD'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8]
            items = [9]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            rf = RandomForestClassifier(n_estimators=95, max_depth=6, min_samples_split=5, max_leaf_nodes=48,
                                        bootstrap='True', random_state=1, max_features=6)
            clf = AdaBoostClassifier(n_estimators=50, base_estimator=rf, learning_rate=1, random_state=1)

        elif (disease == 'Breast Cancer'):
            names = [1, 2, 3, 4, 5, 6]
            items1 = [1, 2, 3, 4, 5]
            items = [6]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            level0 = list()
            level0.append(('svc', SVC(C=0.1, gamma=1, kernel="linear", random_state=1)))
            level0.append(('lr', LogisticRegression(C=1, max_iter=2500, penalty="l1", solver="saga", random_state=1)))
            level1 = LogisticRegression(C=1.623776739188721, max_iter=100, solver='lbfgs', random_state=1)
            clf = sklearn.ensemble.StackingClassifier(estimators=level0, final_estimator=level1)

        elif (disease == 'Cervical Cancer'):  # Done
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8]
            items = [9]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            x_data = x_data.values
            x_class = x_class.values
            clf = xgb.XGBClassifier(booster='gblinear', gamma=0, learning_rate=0.1, max_depth=1, n_estimators=110,
                                    random_state=42)

        elif (disease == 'Multiple Sclerosis'):  # Done
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            items = [12]
            data = pandas.read_csv(savedFile, names=names)
            x_data, y_data, x_class, y_class = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:], test_size=test,
                                                                random_state=0, stratify=data.iloc[:, -1:])
            clf = ExtraTreesClassifier(random_state=0, n_jobs=-1, max_depth=12, max_leaf_nodes=None, n_estimators=450,
                                       min_samples_leaf=1)

        elif (disease == 'Schizophrenia'):  # needs to be updated
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            items = [13]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=101)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2, weights='uniform')

        elif (disease == 'Chronic Obstructive Pulmonary Disease'):  # Done
            names = [1, 2, 3, 4, 5, 6, 7, 8]
            items1 = [1, 2, 3, 4, 5, 6, 7]
            items = [8]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, n_neighbors=3, weights='distance')

        elif (disease == 'Liver Cirrhosis'):  # Done
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            items = [19]
            data = pandas.read_csv(savedFile, names=names)
            x_data, y_data, x_class, y_class = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:], test_size=test,
                                                                random_state=42, stratify=data.iloc[:, -1:])
            x_data = x_data.values
            x_class = x_class.values
            clf = xgb.XGBClassifier(colsample_bytree=1.0, gamma=2, max_depth=4, min_child_weight=1, subsample=0.6,
                                    random_state=42)

        elif (disease == 'Parkinsons Disease'):  # Done
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
            items = [27]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=4)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = lgbm.LGBMClassifier(num_leaves= 248, learning_rate= 0.0772285981543976, subsample_for_bin= 229445, min_child_samples= 16, reg_alpha= 2.1330769743535864e-06,
              reg_lambda= 2.2121431605666793, colsample_bytree= 0.7815836465183598, subsample=0.823756204832638, n_estimators= 72, random_state=42)

        elif (disease == 'Hepatitis C'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            items = [15]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = GradientBoostingClassifier(criterion='squared_error', learning_rate=0.15, max_depth=5, max_features='log2', n_estimators=40, random_state=42)

        elif (disease == 'Depression'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            items = [23]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=123)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            svm_best = SVC(C=7, gamma=0.1, random_state=123, probability=True)
            dt_best = sklearn.tree.DecisionTreeClassifier(max_depth=7, random_state=123)
            knn_best = KNeighborsClassifier()
            clf = VotingClassifier([('svm1', svm_best), ('svm2', svm_best), ('dt1', dt_best), ('dt2', dt_best), ('knn', knn_best)],voting='hard')
        elif (disease == 'Hypertension'):
            names = [1,2,3,4,5,6,7,8,9,10,11]
            items1 = [1,2,3,4,5,6,7,8,9,10]
            items = [11]
            data = pandas.read_csv(savedFile, names=names)
            #convert all attributes to float
            data = data.astype(float)
            train, test = train_test_split(data, test_size=0.2, random_state=123)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = HistGradientBoostingClassifier()
        elif (disease == 'Colorectral Cancer'):
            names= [1,2,3,4,5,6,7,8,9,10,11,12]
            items1 = [1,2,3,4,5,6,7,8,9,10,11]
            items = [12]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=0.2, random_state=123)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = AdaBoostClassifier()
        elif (disease == 'Skin Cancer'):
            names=[1,2,3,4,5,6,7,8,9]
            items1 = [1,2,3,4,5,6,7,8]
            items = [9]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=0.2, random_state=123)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_split=2, random_state=0)
        elif (disease == 'Sickle Cell Anemia'):
            names = [1, 2, 3, 4, 5,6]
            items1 = [1, 2, 3, 4,5]
            items = [6]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, shuffle=True, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = RandomForestClassifier(max_depth= None, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 200) 
            

        elif (disease == 'Epileptic Seizure'):
            names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,28,29,30,31,32,33,34,35,36,37,38]
            items1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,28,29,30,31,32,33,34,34]
            items = [1]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            best_model = LogisticRegression(C= 0.1, solver= 'liblinear')

        elif (disease == 'Osteoporosis'):  
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16]
            items = [17]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test,shuffle=True, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf = RandomForestClassifier(max_depth= None, min_samples_leaf= 1, min_samples_split= 7, n_estimators= 250)

        elif (disease == 'Stroke'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13]
            items = [14]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            
            clf = sklearn.svm.SVC(C=1,  kernel='rbf',gamma='scale')


        elif (disease == 'PCOS'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39,40,41,42]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39,40,41]
            items = [42]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5,
                                       max_features='log2')

        elif (disease == 'Pancreatic Cancer'):
            names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            items = [10]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test, random_state=42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            
            clf = sklearn.svm.SVC(C=2, kernel='rbf', gamma='scale')
        # End first if

        # run the model and save it in .sav format
        clf.fit(x_data, x_class)
        currentDate = datetime.now()
        modelName = currentDate.strftime("%Y-%m-%d %H.%M.%S") + '.sav'

        # save the model name in the database file (dump)
        pickle.dump(clf, open('./static/model/' + modelName, 'wb'))
        loaded_model = pickle.load(open('./static/model/' + modelName, 'rb'))
        result = loaded_model.score(y_data, y_class)
        prd = loaded_model.predict(y_data)

        # save the accuracy with the attibutes used in training and testing in DB
        acc = str(float(result))
        totalInst = str(len(data))
        testInst = str(len(y_data))

        try:
            db, client = connection()
            data = {'ModelName': modelName, 'ModelType': disease, 'TrainingPercent': training, 'Accuracy': acc,
                    'TotalInstances': totalInst, 'TestInstances': testInst}

            db.tempmodel.insert_one(data)
            # conn.commit()
        finally:
            client.close()

        flag = 'T'
        # ctypes.windll.user32.MessageBoxW(0, disease+" model is generated successfully", "", 0)
        flash("Model is Generated Successfully", 'info')
        return redirect(url_for('display'))

# Update Model
@app.route('/updateModel', methods=['POST'])
def updateModel():
    if request.method == 'POST':
        aModelName = request.form['mName']
        aModelName = aModelName[12:]
        try:
            db, client = connection()
            result = db.tempmodel.find({"ModelName": aModelName},
                                       {"_id": 0, "ModelType": 1, "TrainingPercent": 1, "Accuracy": 1,
                                        "TotalInstances": 1, "TestInstances": 1})

            if result is None:
                session['updateM'] = True
                return redirect(url_for('display'))

            aDisease = result[0]['ModelType']
            aTraining = int(result[0]['TrainingPercent'])
            aAcc = float(result[0]['Accuracy'])
            aTotalInst = int(result[0]['TotalInstances'])
            aTestInst = int(result[0]['TestInstances'])

            db.model.update_many({'DiseaseType': aDisease}, {'$set': {'Active': 0}})

            model_document = {"ModelName": aModelName,
                              "DiseaseType": aDisease,
                              "Accuracy": aAcc,
                              "TotalInstances": aTotalInst,
                              "TestInstances": aTestInst,
                              "TrainingPercent": aTraining,
                              "Active": 1}

            db.model.insert_one(model_document)

            result2 = db.tempmodel.find({"ModelType": aDisease, "ModelName": {"$ne": aModelName}}, {"ModelName": 1})
            for row in result2:
                os.remove(os.path.join(app.config['UPLOADED_ITEMS_DEST'], row["ModelName"]))

            db.tempmodel.delete_many({"ModelType": aDisease})


        finally:
            client.close()
        session['updateM'] = True
        return redirect(url_for('display'))

def send_confirmation_email(user_email, name, html):  # funtion that sends email confirmtion email with token to confirm email address
    confirm_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    # create the token to confirm email address
    confirm_url = url_for(
        'confirm_email',
        token=confirm_serializer.dumps(user_email, salt='email-confirmation-salt'),
        _external=True)
    # split the username to first and last name
    FLname = name.split(" ")
    # render the change_email_confirmation html template with the user first name and token  
    html = render_template(html, confirm_url=confirm_url, user=FLname[0])
    # send the email to user with the change_email_confirmation html template
    mail.send(email('Confirm Your Email Address', user_email, html))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(405)
def page_not_found(e):
    return render_template('404.html'), 404

class Email(Form):
    uemail = StringField('Email', [DataRequired(),
                                   validators.Email()])

@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    form = Email(request.form)

    if request.method == 'GET':
        return render_template('forgot.html', form=form)

    elif request.method == 'POST' and form.validate():
        uemail = form.uemail.data.lower()
        db, client = connection()

        user_data = db.account.find_one({'Email': uemail})

        if user_data is not None:
            username = user_data['_id']
            name = user_data['Name']
            useremail = user_data['Email'].lower()
            emailfalg = user_data['EmailConfirmed']

            if emailfalg == 1:
                tpassword = password_gen()
                temppass = sha256_crypt.encrypt(str(tpassword))

                db.account.find_one_and_update({'_id': username}, {'$set': {'TempPassFlag': 1, 'Password': temppass}})
                flash("You can enter your Account now")
                FLname = name.split(" ")
                html = render_template('forgetpass_email.html', user=FLname[0], temp=tpassword, form=uemail)
                msg = email("Reset password", useremail, html)
                mail.send(msg)
            else:
                error = "Email unconfirmed, please confirm your new email address (link sent to you in email)."
                send_confirmation_email(useremail, name, 'change_email_confirmation.html')
                return render_template("forgot.html", form=form, error=error)
            form1 = Login(request.form)
            return render_template("login.html", form=form1)
        #
        else:
            error = "Email does not exist"
            return render_template("forgot.html", form=form, error=error)
    else:
        return render_template("forgot.html", form=form)

class Login(Form):
    ID = StringField('Username')
    password = PasswordField('Password')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = Login(request.form)
    if request.method == 'GET':
        return render_template('login.html', form=form)

    elif request.method == 'POST' and form.validate():
        ID = form.ID.data
        password = form.password.data
        if ID.strip() == "" or password.strip() == "":
            error = "All fields must be filled"
            return render_template('login.html', form=form, error=error)

        db, client = connection()

        data = db.account.find_one({'_id': ID}, {'Password': 1, 'Role': 1})

        if data is None:
            error = "Invalid username or password"
            return render_template('login.html', form=form, error=error)

        else:
            dbpassword = data['Password']

            v = sha256_crypt.verify(password, dbpassword)
            if (v == True):

                if (data['Role'] == 'admin'):
                    session['logged_in'] = True
                    session['username'] = ID
                    session['role'] = data['Role']
                    return redirect(url_for('profile'))

                elif (data['Role'] == 'registered user'):
                    session['logged_in'] = True
                    session['username'] = ID
                    session['role'] = data['Role']
                    return redirect(url_for('profile'))

                elif (data['Role'] == 'medical specialist'):
                    session['logged_in'] = True
                    session['username'] = ID
                    session['role'] = data['Role']
                    return redirect(url_for('profile'))

                elif (data['Role'] == 'laboratory specialist'):
                    session['logged_in'] = True
                    session['username'] = ID
                    session['role'] = data['Role']
                    return redirect(url_for('profile'))

            else:
                return render_template('login.html', error="Invalid username or password", form=form)

    elif request.method == 'POST':
        error = "Invalid username or password"
        return render_template('login.html', form=form)

def calculate_age(born):
    today = date.today()
    try:
        birthday = born.replace(year=today.year)
    except ValueError:
        birthday = born.replace(year=today.year, month=born.month + 1, day=1)
    if birthday > today:
        return today.year - born.year - 1
    else:
        return today.year - born.year

@app.route('/idcheck', methods=['POST'])
def IDcheck():
    try:
        db, client = connection()

        PID = int(request.form['MRN'])  # PID==MRN
        count = db.patient.count_documents({'_id': PID})

        if len(str(PID)) > 9 or len(str(PID)) < 7:
            return render_template("MSdiagnose.html", PID_len='False')

        if count > 1:
            flash('Too many patients with one MRN', 'error')
        elif count < 1:
            flash('No patient exists with this MRN', 'error')
        else:
            patient_info = db.patient.find({'_id': PID}, {'birth_date': 1, 'gender': 1, 'username': 1})
            username = patient_info[0]['username']

            name = db.account.find({'_id': username}, {'Name': 1})
            return render_template("MSdiagnose.html", patient="True", add="False", PID=PID, name=name[0]['Name'],
                                   gender=str(patient_info[0]['gender']), bdate=patient_info[0]['birth_date'])
    finally:
        client.close()

    return render_template("MSdiagnose.html", patient="False", PID=PID)

@app.route('/addPatient', methods=['POST'])
def addpatient():
    PID = int(request.form['NUsername'])  # New Patient MRN "Hidden Field"
    if len(str(PID)) > 9 or len(str(PID)) < 7:
        return render_template("MSdiagnose.html", PID_len='False')

    client = None
    try:
        email = str(request.form['email']).lower()
        dbname = str(request.form['name'])
        dbbdate = str(request.form['birthdate'])
        dbgender = str(request.form['gender'])
        username = email.split('@')[0]

        db, client = connection()

        dataem = db.account.find_one({'Email': email})
    finally:
        if client:
            client.close()

    if dataem is not None:
        flash("The email already exists in another account", 'error')
        return render_template("MSdiagnose.html", patient="False", PID=PID, email='True')
    else:
        try:
            db, client = connection()
            exist = db.account.find_one({'_id': username})

            if exist is not None:
                username = username_gen(username.lower())

            db.account.insert_one({
                '_id': username,
                'Name': dbname,
                'Email': email,
                'Password': sha256_crypt.hash(password_gen()),
                'TempPassFlag': 0,
                'Role': 'registered user',
                'EmailConfirmed': 0 
            })
            db.patient.insert_one({
                '_id': PID,
                'username': username,
                'birth_date': dbbdate,
                'gender': dbgender
            })

            send_confirmation_email(email, dbname, 'createaccount_email.html')

            return render_template("MSdiagnose.html", patient="True", add="True", PID=PID,
                                   name=dbname, email=email, gender=dbgender, bdate=dbbdate)
        except Exception as e:
            print("Error occurred:", str(e))
        finally:
            if client:
                client.close()
    return render_template("MSdiagnose.html", patient="False", PID=PID)



# .................................................
# ...........ADD THE DISEASE METHOD HERE ..........
# .................................................

@app.route('/diagnosis')
def dmDiagnosis():

    Role = ""
    if 'role' in session:
        Role = session['role']

    if Role == 'medical specialist':
        return render_template("MSdiagnose.html")
    if Role == 'registered user':
        return render_template("RUdiagnosis.html")
    if Role == 'laboratory specialist':
        return render_template("LSdiagnosis.html")
    if Role == 'admin':
        return render_template("404.html")
    return render_template("diagnose.html")

@app.route('/LaboratorySpecialistsDiagnose', methods=['GET', 'POST'])
def labDiagnosis():
    if request.method == 'POST':
        checked = request.form.getlist('disease')
        features = []
        results = []

        if 'Chronic Kidney Disease' in checked:
            bun = request.form['LS_BUN']
            creatinine = request.form['LS_Creatinine']

            features.append({'bun': bun, 'creatinine': creatinine})
            ckd = ckddiagnosis()
            results.append(ckd)

        if 'Diabetes Mellitus' in checked:
            sugar_level = request.form['LS_SugarLevel']
            hematocrit = request.form['LS_Hematocrit']
            mpv = request.form['LS_MPV']

            features.append({'sugar_level': sugar_level, 'hematocrit': hematocrit, 'mpv': mpv})
            diabetes = dmdiagnosis()
            results.append(diabetes)

        if 'Coronary Heart Disease' in checked:
            gender = request.form['LS_Sex']
            age = request.form['LS_Age']
            rdw = request.form['LS_RDW']
            platelet = request.form['LS_Platelet']
            mpv = request.form['LS_MPV']
            mono = request.form['LS_MONO']
            basophil_instrument = request.form['LS_Basophil_Instrument']
            basophil_instrument_abs = request.form['LS_Basophil_Instrument_Abs']
            potassium = request.form['LS_Potassium']
            ggtp = request.form['LS_GGTP']
            sgpt = request.form['LS_SGPT']
            sgot = request.form['LS_SGOT']
            neugran = request.form['LS_NeuGran']
            neugran_abs = request.form['LS_NeuGran_Abs']
            anion_gap = request.form['LS_Anion_Gap']

            features.append({'gender': gender, 'age': age, 'rdw': rdw, 'platelet': platelet,
                             'mpv': mpv, 'mono': mono, 'basophil_instrument': basophil_instrument,
                             'basophil_instrument_abs': basophil_instrument_abs, 'potassium': potassium,
                             'ggtp': ggtp, 'sgpt': sgpt, 'sgot': sgot,
                             'neugran': neugran, 'neugran_abs': neugran_abs, 'anion_gap': anion_gap})

            chd = CHDDdiagnosis()
            results.append(chd)

        if 'Rheumatoid Arthritis Disease' in checked:
            gender = request.form['LS_Sex']
            age = request.form['LS_Age']
            albumin = request.form['LS_Albumin']
            alk_phos = request.form['LS_Alk_Phos']
            bun = request.form['LS_BUN']
            chloride = request.form['LS_Chloride']
            carbon_dioxide = request.form['LS_Carbon_Dioxide']
            creatinine = request.form['LS_Creatinine']
            direct_bilirubin = request.form['LS_DirectBilirubin']
            ggpt = request.form['LS_GGTP']
            hemoglobin = request.form['LS_Hemoglobin']
            hematocrit = request.form['LS_Hematocrit']
            potassium = request.form['LS_Potassium']
            lad = request.form['LS_LAD']
            mch = request.form['LS_MCH']
            mchc = request.form['LS_MCHC']
            mcv = request.form['LS_MCV']
            mpv = request.form['LS_MPV']
            sodium = request.form['LS_Sodium']
            platelet = request.form['LS_Platelet']
            rbc_count = request.form['LS_RBCcount']
            rdw = request.form['LS_RDW']
            sgot = request.form['LS_SGOT']
            sgpt = request.form['LS_SGPT']
            total_bilirubin = request.form['LS_TotalBilirubin']
            total_protein = request.form['LS_TotalProtein']

            features.append({
                'gender': gender, 'age': age, 'albumin': albumin, 'alk_phos': alk_phos, 'bun': bun,
                'chloride': chloride, 'carbon_dioxide': carbon_dioxide, 'creatinine': creatinine,
                'direct_bilirubin': direct_bilirubin, 'ggpt': ggpt,
                'hemoglobin': hemoglobin, 'hematocrit': hematocrit, 'potassium': potassium, 'lad': lad, 'mch': mch,
                'mchc': mchc, 'mcv': mcv, 'mpv': mpv, 'sodium': sodium, 'platelet': platelet, 'rbc_count': rbc_count,
                'rdw': rdw, 'sgot': sgot, 'sgpt': sgpt, 'total_bilirubin': total_bilirubin,
                'total_protein': total_protein
            })
            ra = RAdiagnosis()
            results.append(ra)

        if 'Asthma Disease' in checked:
            gender = request.form['LS_Sex']
            age = request.form['LS_Age']
            basophil_instrument = request.form['LS_Basophil_Instrument']
            hematocrit = request.form['LS_Hematocrit']
            hemoglobin = request.form['LS_Hemoglobin']
            mch = request.form['LS_MCH']
            mchc = request.form['LS_MCHC']
            mpv = request.form['LS_MPV']
            wbc_count = request.form['LS_WBCcount']

            features.append({'gender': gender, 'age': age, 'basophil_instrument': basophil_instrument,
                             'hematocrit': hematocrit, 'hemoglobin': hemoglobin, 'mch': mch,
                             'mchc': mchc, 'mpv': mpv, 'wbc_count': wbc_count})

            asthma = asdiagnosis()
            results.append(asthma)

        if 'Thyroid Cancer' in checked:
            gender = request.form['LS_Sex']
            age = request.form['LS_Age']
            hematocrit = request.form['LS_Hematocrit']
            mchc = request.form['LS_MCHC']
            mpv = request.form['LS_MPV']
            rbc_count = request.form['LS_RBCcount']
            wbc_count = request.form['LS_WBCcount']

            features.append({'gender': gender, 'age': age, 'hematocrit': hematocrit, 'mchc': mchc, 'mpv': mpv,
                             'rbc_count': rbc_count, 'wbc_count': wbc_count})

            thyroid = tcdiagnosis()
            results.append(thyroid)

        if 'Schizophrenia' in checked:
            age = request.form['LS_Age']
            ggt = request.form['LS_GGT']
            platelet = request.form['LS_Platelet']

            features.append({'age': age, 'ggt': ggt, 'platelet': platelet})

            schizophrenia = Schizodiagnosis()
            results.append(schizophrenia)

        if 'Hypothyroidism' in checked:
            age = request.form['LS_Age']
            bp_systolic = request.form['LS_BP_Systolic']
            respiratory_rate = request.form['LS_RespiratoryRate']
            mcv = request.form['LS_MCV']
            pulse = request.form['LS_Pulse']

            features.append({'age': age, 'bp_systolic': bp_systolic, 'respiratory_rate': respiratory_rate,
                             'mcv': mcv, 'pulse': pulse})

            hypothyroidism = Hypodiagnosis()
            results.append(hypothyroidism)

        if 'Prostate Cancer' in checked:
            pc_perimeter = request.form['LS_PCperimeter']
            pc_area = request.form['LS_PCarea']
            pc_smoothness = request.form['LS_PCsmoothness']
            pc_compactness = request.form['LS_PCcompactness']

            features.append({'pc_perimeter': pc_perimeter, 'pc_area': pc_area, 'pc_smoothness': pc_smoothness,
                             'pc_compactness': pc_compactness})

            pc = PCdiagnosis()
            results.append(pc)

        if 'Multiple Sclerosis' in checked:
            age = request.form['LS_Age']
            alt = request.form['LS_ALT']
            ldh = request.form['LS_LDH']
            creatinine = request.form['LS_Creatinine']
            bun = request.form['LS_BUN']
            total_bilirubin = request.form['LS_TotalBilirubin']
            ggt = request.form['LS_GGT']
            alk_pho = request.form['LS_Alk_Phos']
            ast = request.form['LS_AST']
            platelet = request.form['LS_Platelet']
            bp_systolic = request.form['LS_BP_Systolic']

            features.append({'age': age, 'alt': alt, 'ldh': ldh, 'creatinine': creatinine, 'bun': bun,
                             'total_bilirubin': total_bilirubin, 'ggt': ggt, 'alk_pho': alk_pho, 'ast': ast,
                             'platelet': platelet, 'bp_systolic': bp_systolic})

            ms = MSdiagnosis()
            results.append(ms)

        if 'Alzheimer’s Disease' in checked:
            sex = request.form['LS_Sex']
            age = request.form['LS_Age']
            pulse = request.form['LS_Pulse']
            respiratory_rate = request.form['LS_RespiratoryRate']
            bp_diastolic = request.form['LS_BP_Diastolic']
            wbc_count = request.form['LS_WBCcount']
            rbc_count = request.form['LS_RBCcount']
            hemoglobin = request.form['LS_Hemoglobin']
            hematocrit = request.form['LS_Hematocrit']
            mcv = request.form['LS_MCV']
            mch = request.form['LS_MCH']
            rdw = request.form['LS_RDW']
            mpv = request.form['LS_MPV']

            features.append({'gender': sex, 'age': age, 'pulse': pulse, 'respiratory_rate': respiratory_rate,
                             'bp_diastolic': bp_diastolic, 'wbc_count': wbc_count, 'rbc_count': rbc_count,
                             'hemoglobin': hemoglobin, 'hematocrit': hematocrit, 'mcv': mcv, 'mch': mch,
                             'rdw': rdw, 'mpv': mpv})

            ad = ADdiagnosis()
            results.append(ad)

        if 'Lung Cancer' in checked:
            gender = request.form['LS_Sex']
            age = request.form['LS_Age']
            smoking = request.form['LS_smoking']
            yellow_fingers = request.form['LS_yellowFingers']
            anxiety = request.form['LS_anxiety']
            wheezing = request.form['LS_wheezing']
            peer_pressure = request.form['LS_peerPressure']
            chronic_disease = request.form['LS_chronicDisease']
            fatigue = request.form['LS_fatigue']
            allergy = request.form['LS_allergy']
            coughing = request.form['LS_coughing']
            alcohol = request.form['LS_alcohol']
            shortness_of_breath = request.form['LS_shortness_of_Breath']
            swallowing_difficulty = request.form['LS_swallowing_Difficulty']
            chest_pain = request.form['LS_chest_Pain']

            features.append({'gender': gender, 'age': age, 'smoking': smoking, 'yellow_fingers': yellow_fingers,
                             'anxiety': anxiety, 'wheezing': wheezing, 'peer_pressure': peer_pressure,
                             'chronic_disease': chronic_disease, 'fatigue': fatigue, 'allergy': allergy,
                             'coughing': coughing, 'alcohol': alcohol, 'shortness_of_breath': shortness_of_breath,
                             'swallowing_difficulty': swallowing_difficulty, 'chest_pain': chest_pain})

            lc = LCdiagnosis()
            results.append(lc)

        if 'Glaucoma' in checked:
            at = request.form['LS_at']
            ean = request.form['LS_ean']
            mhci = request.form['LS_mhci']
            vasi = request.form['LS_vasi']
            varg = request.form['LS_varg']
            vars = request.form['LS_vars']
            tmi = request.form['LS_tmi']

            features.append({'at': at, 'ean': ean, 'mhci': mhci, 'vasi': vasi, 'varg': varg, 'vars': vars, 'tmi': tmi})

            glaucoma = Glaucomadiagnosis()
            results.append(glaucoma)

        if 'Liver Cirrhosis' in checked:
            gender = request.form['LS_Sex']
            age = request.form['LS_Age']
            ndays = request.form['LS_NDays']
            hepatomegaly = request.form['LS_Hepatomegaly']
            spiders = request.form['LS_Spiders']
            edema = request.form['LS_Edema']
            cholesterol = request.form['LS_Cholesterol']
            copper = request.form['LS_Copper']
            sgot = request.form['LS_SGOT']
            platelet = request.form['LS_Platelet']
            prothrombin = request.form['LS_Prothrombin']
            ascites = request.form['LS_Ascites']
            bilirubin = request.form['LS_SerumBilirubin']
            albumin = request.form['LS_Albumin']
            alk_phos = request.form['LS_Alk_Phos']
            triglycerides = request.form['LS_Triglycerides']
            drug = request.form['LS_Drug']
            status_lc = request.form['LS_StatusLC']

            features.append({'gender': gender, 'age': age, 'ndays': ndays, 'hepatomegaly': hepatomegaly,
                             'spiders': spiders, 'edema': edema, 'cholesterol': cholesterol, 'copper': copper,
                             'sgot': sgot, 'platelet': platelet, 'prothrombin': prothrombin, 'ascites': ascites,
                             'bilirubin': bilirubin, 'albumin': albumin, 'alk_phos': alk_phos,
                             'triglycerides': triglycerides,
                             'drug': drug, 'status_lc': status_lc})

            liver_cirrhosis = LCHdiagnosis()
            results.append(liver_cirrhosis)

        if 'Parkinson’s Disease' in checked:
            gender = request.form['LS_Sex']
            age = request.form['LS_Age']
            anion_gap = request.form['LS_Anion_Gap']
            alt = request.form['LS_ALT']
            ldh = request.form['LS_LDH']
            wbc = request.form['LS_WBC']
            rbc = request.form['LS_RBC']
            hemoglobin = request.form['LS_Hemoglobin']
            hematocrit = request.form['LS_Hematocrit']
            sodium = request.form['LS_Sodium']
            potassium = request.form['LS_Potassium']
            chloride = request.form['LS_Chloride']
            carbon_dioxide = request.form['LS_Carbon_Dioxide']
            creatinine = request.form['LS_Creatinine']
            total_protein = request.form['LS_TotalProtein']
            albumin = request.form['LS_Albumin']
            bun = request.form['LS_BUN']
            total_bilirubin = request.form['LS_TotalBilirubin']
            direct_bilirubin = request.form['LS_DirectBilirubin']
            ggt = request.form['LS_GGT']
            mcv = request.form['LS_MCV']
            mch = request.form['LS_MCH']
            mchc = request.form['LS_MCHC']
            alk_phos = request.form['LS_Alk_Phos']
            rdw = request.form['LS_RDW']
            ast = request.form['LS_AST']

            features.append({'gender': gender, 'age': age, 'anion_gap': anion_gap, 'alt': alt, 'ldh': ldh, 'wbc': wbc,
                             'rbc': rbc, 'hemoglobin': hemoglobin, 'hematocrit': hematocrit, 'sodium': sodium,
                             'potassium': potassium, 'chloride': chloride, 'carbon_dioxide': carbon_dioxide,
                             'creatinine': creatinine, 'total_protein': total_protein, 'albumin': albumin,
                             'bun': bun, 'total_bilirubin': total_bilirubin, 'direct_bilirubin': direct_bilirubin,
                             'ggt': ggt, 'mcv': mcv, 'mch': mch, 'mchc': mchc, 'alk_phos': alk_phos, 'rdw': rdw,
                             'ast': ast})

            pd = PRDdiagnosis()
            results.append(pd)

        if 'Cervical Cancer' in checked:
            num_of_diag = request.form['LS_NumOfDiag']
            condylomatosis = request.form['LS_Condylomatosis']
            std_syphilis = request.form['LS_StdSyphilis']
            hiv = request.form['LS_HIV']
            std_hpv = request.form['LS_STD_HPV']
            dx = request.form['LS_Dx']
            dxcin = request.form['LS_DxCIN']
            dxhpv = request.form['LS_DxHPV']

            features.append({'num_of_diag': num_of_diag, 'condylomatosis': condylomatosis, 'std_syphilis': std_syphilis,
                             'hiv': hiv, 'std_hpv': std_hpv, 'dx': dx, 'dxcin': dxcin, 'dxhpv': dxhpv})

            cc = CervicalCancerdiagnosis()
            results.append(cc)

        if 'Hepatitis C' in checked:
            age = request.form['LS_Age']
            total_protein = request.form['LS_TotalProtein']
            total_bilirubin = request.form['LS_TotalBilirubin']
            direct_bilirubin = request.form['LS_DirectBilirubin']
            ggt = request.form['LS_GGT']
            alk_phos = request.form['LS_Alk_Phos']
            lymphocyte = request.form['LS_Lymphocyte']
            neu_gran_abs = request.form['LS_NeuGran_Abs']
            platelet = request.form['LS_Platelet']
            basophil_inst = request.form['LS_Basophil_Instrument']
            bp_systolic = request.form['LS_BP_Systolic']
            fall_risk = request.form['LS_FallRiskMorse']
            body_mass = request.form['LS_BodyMass']
            int_norm_rati = request.form['LS_IntNormRati']

            features.append({'age': age, 'total_protein': total_protein, 'total_bilirubin': total_bilirubin,
                             'direct_bilirubin': direct_bilirubin, 'ggt': ggt, 'alk_phos': alk_phos,
                             'lymphocyte': lymphocyte, 'neu_gran_abs': neu_gran_abs, 'platelet': platelet,
                             'basophil_inst': basophil_inst,
                             'bp_systolic': bp_systolic, 'fall_risk': fall_risk, 'body_mass': body_mass,
                             'int_norm_rati': int_norm_rati})

            hc = HepCdiagnosis()
            results.append(hc)

        if 'Depression' in checked:
            age = request.form['LS_Age']
            hh_size = request.form['LS_hhsize']
            edu_level = request.form['LS_EducationLevel']
            val_livestock = request.form['LS_ValOfLivestock']
            val_durab_good = request.form['LS_ValOfDurabGood']
            val_saving = request.form['LS_ValSaving']
            land_owned = request.form['LS_LandOwned']
            alcohol_consumed = request.form['LS_AlcoholConsumed']
            tobacco_consumed = request.form['LS_TobaccoConsumed']
            edu_expenditure = request.form['LS_EduExpenditure']
            ent_nonag_flowcost = request.form['LS_Ent_nonag_flowcost']
            ent_animalstockrev = request.form['LS_Ent_animalstockrev']
            ent_total_cost = request.form['LS_Ent_total_cost']
            fs_adwholed_often = request.form['LS_Fs_adwholed_often']
            nondurable_investment = request.form['LS_Nondurable_investment']
            amount_received_mpesa = request.form['LS_Amount_received_mpesa']
            married = request.form['LS_Married']
            children = request.form['LS_Children']
            hh_children = request.form['LS_hh_children']
            ent_nonagbusiness = request.form['LS_Ent_nonagbusiness']
            saved_mpesa = request.form['LS_Saved_mpesa']
            early_survey = request.form['LS_Early_survey']

            features.append({'age': age, 'hh_size': hh_size, 'edu_level': edu_level, 'val_livestock': val_livestock,
                             'val_durab_good': val_durab_good, 'val_saving': val_saving, 'land_owned': land_owned,
                             'alcohol_consumed': alcohol_consumed, 'tobacco_consumed': tobacco_consumed,
                             'edu_expenditure': edu_expenditure, 'ent_nonag_flowcost': ent_nonag_flowcost,
                             'ent_animalstockrev': ent_animalstockrev, 'ent_total_cost': ent_total_cost,
                             'fs_adwholed_often': fs_adwholed_often, 'nondurable_investment': nondurable_investment,
                             'amount_received_mpesa': amount_received_mpesa, 'married': married, 'children': children,
                             'hh_children': hh_children, 'ent_nonagbusiness': ent_nonagbusiness,
                             'saved_mpesa': saved_mpesa, 'early_survey': early_survey})

            depression = Depressiondiagnosis()
            results.append(depression)

        if 'Chronic Obstructive Pulmonary Disease' in checked:
            sex = request.form['LS_Sex']
            age = request.form['LS_Age']
            smoking = request.form['LS_smoking']
            imagery_part_min = request.form['LS_Imagery_part_min']
            imagery_part_avg = request.form['LS_Imagery_part_avg']
            real_part_min = request.form['LS_Real_part_min']
            real_part_avg = request.form['LS_Real_part_avg']

            features.append({'sex': sex, 'age': age, 'smoking': smoking, 'imagery_part_min': imagery_part_min,
                             'imagery_part_avg': imagery_part_avg, 'real_part_min': real_part_min,
                             'real_part_avg': real_part_avg})

            copd = COPDdiagnosis()
            results.append(copd)

        if 'Hypertension Disease' in checked:
            age = request.form['LS_Age']
            protein= request.form['LS_ProtienUA']
            glucose = request.form['LS_Glucose']
            hemoglobin = request.form['LS_Hemoglobin']
            hdl= request.form['LS_HDL']
            ldl = request.form['LS_HDL_LDL']
            cholesterol = request.form['LS_Cholesterol']
            ldl_chol = request.form['LS_LDL_Cholesterol']
            pH= request.form['LS_pH_Urine']
            calcium = request.form['LS_Calcium']

            features.append({'age': age, 'protein': protein, 'glucose': glucose, 'hemoglobin': hemoglobin,
                             'hdl': hdl, 'ldl': ldl, 'cholesterol': cholesterol, 'ldl_chol': ldl_chol,
                             'pH': pH, 'calcium': calcium})
            
            hypertension = Hyperdiagnosis()
            results.append(hypertension)

        if 'Skin Cancer' in checked:
            age = request.form['LS_Age']
            sex = request.form['LS_Sex']
            smoke= request.form['LS_smoking']
            pesticide = request.form['LS_pesticide']
            skin_cancer_history = request.form['LS_skin_cancer_hist']
            cancer_history	= request.form['LS_cancer_hist']
            has_piped_water = request.form['LS_has_piped_water']
            has_sewage_system = request.form['LS_has_sewage_system']

            features.append({'smoke': smoke,'age': age, 'pesticide': pesticide, 'sex': sex,'skin_cancer_history': skin_cancer_history,
                            'cancer_history': cancer_history, 'has_piped_water': has_piped_water, 'has_sewage_system': has_sewage_system})
            
            SkinCancer = SkinDiagnosis()
            results.append(SkinCancer)
            
            
        if 'Colorectal Cancer' in checked:
            age = request.form['LS_Age']
            p16540= request.form['LS_p16540']
            p16580 = request.form['LS_p16580']
            mdm2 = request.form['LS_mdm2']
            GAL3 = request.form['LS_GAL3']
            TIM1 = request.form['LS_TIM1']
            
            features.append({'age': age,'p16540': p16540,'p16580': p16580, 'mdm2': mdm2, 'GAL3': GAL3,'TIM1': TIM1})
            
            ColorectCancer = ColorectDiagnosis()
            results.append(ColorectCancer)
            # Basem Team
        if 'Pancreatic_Cancer' in checked:
            age = int(request.form['LS_Age'])
            plasma = float(request.form['plasma'])
            creatinine = float(request.form['creatinine'])
            lyve1 = float(request.form['lyve1'])
            reg1b = float(request.form['reg1b'])
            tff1 = float(request.form['tff1'])
            rega1a = float(request.form['rega1a'])
            sex_m = 1 if request.form['LS_Sex'] == 'Male' else 0
            sex_f = 1 if request.form['LS_Sex'] == 'Female' else 0

            features.append({'age': age, 'plasma': plasma, 'creatinine': creatinine, 'lyve1': lyve1,
                            'reg1b': reg1b, 'tff1': tff1, 'rega1a': rega1a, 'sex_m': sex_m, 'sex_f': sex_f})

            disease = pc2diagnosis()  # Replace 'DiseaseDiagnosis' with the appropriate class name
            results.append(disease)

        if 'Stroke' in checked:
            gender = 1 if request.form['LS_Sex'] == 'male' else 0
            age = int(request.form['LS_Age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart disease'])
            ever_married = int(request.form['ever married'])
            work_type = int(request.form['work type'])
            residence_type = int(request.form['residence type'])
            avg_glucose_level = int(request.form['avg glucose level'])
            bmi = int(request.form['bmi'])
            smoking_status = int(request.form['smoking status'])
            age_category = int(request.form['age category'])
            glucose_category = int(request.form['glucose category'])
            bmi_category = int(request.form['bmi category'])

            features.append({
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': ever_married,
                'work_type': work_type,
                'residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status,
                'age_category': age_category,
                'glucose_category': glucose_category,
                'bmi_category': bmi_category
            })

            disease = strokediagnosis()  # Replace 'DiseaseDiagnosis' with the appropriate class name
            results.append(disease)

        if 'PCOS' in checked:
            age = int(request.form['LS_Age'])
            weight = float(request.form['Weight (Kg)'])
            height = float(request.form['Height(Cm)'])
            bmi = float(request.form['BMI'])
            blood_group = request.form['Blood Group']
            pulse_rate = float(request.form['Pulse rate(bpm)'])
            rr = float(request.form['RR (breaths/min)'])
            hb = float(request.form['Hb(g/dl)'])
            cycle_ri = int(request.form['Cycle(R/I)'] == 'R')
            cycle_length = int(request.form['Cycle length(days)'])
            marriage_status = int(request.form['Marraige Status (Yrs)'])
            pregnant = int(request.form['Pregnant(Y/N)'] == 'Y')
            num_aborptions = int(request.form['No. of aborptions'])
            beta_hcg_i = float(request.form['  I   beta-HCG(mIU/mL)'])
            beta_hcg_ii = float(request.form['II    beta-HCG(mIU/mL)'])
            fsh = float(request.form['FSH(mIU/mL)'])
            lh = float(request.form['LH(mIU/mL)'])
            fsh_lh = float(request.form['FSH/LH'])
            hip = float(request.form['Hip(inch)'])
            waist = float(request.form['Waist(inch)'])
            waist_hip_ratio = float(request.form['Waist:Hip Ratio'])
            tsh = float(request.form['TSH (mIU/L)'])
            amh = float(request.form['AMH(ng/mL)'])
            prl = float(request.form['PRL(ng/mL)'])
            vit_d3 = float(request.form['Vit D3 (ng/mL)'])
            prg = float(request.form['PRG(ng/mL)'])
            rbs = float(request.form['RBS(mg/dl)'])
            weight_gain = int(request.form['Weight gain(Y/N)'] == 'Y')
            hair_growth = int(request.form['hair growth(Y/N)'] == 'Y')
            skin_darkening = int(request.form['Skin darkening (Y/N)'] == 'Y')
            hair_loss = int(request.form['Hair loss(Y/N)'] == 'Y')
            pimples = int(request.form['Pimples(Y/N)'] == 'Y')
            fast_food = int(request.form['Fast food (Y/N)'] == 'Y')
            reg_exercise = int(request.form['Reg.Exercise(Y/N)'] == 'Y')
            bp_systolic = float(request.form['BP _Systolic (mmHg)'])
            bp_diastolic = float(request.form['BP _Diastolic (mmHg)'])
            follicle_no_l = int(request.form['Follicle No. (L)'])
            follicle_no_r = int(request.form['Follicle No. (R)'])
            avg_f_size_l = float(request.form['Avg. F size (L) (mm)'])
            avg_f_size_r = float(request.form['Avg. F size (R) (mm)'])
            endometrium = float(request.form['Endometrium (mm)'])

            features.append({
                'age': age,
                'weight': weight,
                'height': height,
                'bmi': bmi,
                'blood_group': blood_group,
                'pulse_rate': pulse_rate,
                'rr': rr,
                'hb': hb,
                'cycle_ri': cycle_ri,
                'cycle_length': cycle_length,
                'marriage_status': marriage_status,
                'pregnant': pregnant,
                'num_aborptions': num_aborptions,
                'beta_hcg_i': beta_hcg_i,
                'beta_hcg_ii': beta_hcg_ii,
                'fsh': fsh,
                'lh': lh,
                'fsh_lh': fsh_lh,
                'hip': hip,
                'waist': waist,
                'waist_hip_ratio': waist_hip_ratio,
                'tsh': tsh,
                'amh': amh,
                'prl': prl,
                'vit_d3': vit_d3,
                'prg': prg,
                'rbs': rbs,
                'weight_gain': weight_gain,
                'hair_growth': hair_growth,
                'skin_darkening': skin_darkening,
                'hair_loss': hair_loss,
                'pimples': pimples,
                'fast_food': fast_food,
                'reg_exercise': reg_exercise,
                'bp_systolic': bp_systolic,
                'bp_diastolic': bp_diastolic,
                'follicle_no_l': follicle_no_l,
                'follicle_no_r': follicle_no_r,
                'avg_f_size_l': avg_f_size_l,
                'avg_f_size_r': avg_f_size_r,
                'endometrium': endometrium
                
            })

            disease = pcosdiagnosis()  # Replace 'DiseaseDiagnosis' with the appropriate class name
            results.append(disease)

            if 'Sickle Cell Anemia' in checked:
            
                hemoglobin = request.form['LS_Hemoglobin'] 
                PCV = request.form['LS_PCV']
                rbc_count = request.form['LS_RBCcount'] 
                mcv = request.form['LS_MCV'] 
                mchc = request.form['LS_MCHC']
             
            

            features.append({
                             'hemoglobin': hemoglobin, 'PCV': PCV, 'rbc_count': rbc_count,   
                             'mcv': mcv, 
                             'mchc': mchc})

            scad = SCAdiagnosis()
            results.append(scad)


        if 'EpilepticSeizure' in checked:
                    sex = request.form['LS_Sex']
                    non_psychComorbidities = request.form['LS_non_psychComorbidities']
                    PriorAEDs = request.form['LS_PriorAEDs']
                    AsthmaAttr = request.form['LS_AsthmaAttr']
                    Migraine = request.form['LS_Migraine']
                    ChronicPain = request.form['LS_ChronicPain']
                    DiabetesAttr = request.form['LS_DiabetesAttr']
                    non_metastaticCancer = request.form['LS_non_metastaticCancer']
                    NumberOfNoN_seizureNon_psychMedication = request.form['LS_NumberOfNoN_seizureNon_psychMedication']
                    CurrentAEDs = request.form['LS_CurrentAEDs']
                    Baseline_szFreq = request.form['LS_Baseline_szFreq']
                    MedianDurationOfSeizures = request.form['LS_MedianDurationOfSeizures']
                    NumberOfSeizureTypes = request.form['LS_NumberOfSeizureTypes']
                    InjuryWithSeizure = request.form['LS_InjuryWithSeizure']
                    Catamenial = request.form['LS_Catamenial']
                    TriggerOfSleepDeprivation = request.form['LS_TriggerOfSleepDeprivation']
                    Aura = request.form['LS_Aura']
                    IctalEyeClosure = request.form['LS_IctalEyeClosure']
                    IctalHallucinations = request.form['LS_IctalHallucinations']
                    OralAutomatisms = request.form['LS_OralAutomatisms']
                    Incontinence  = request.form['LS_Incontinence']
                    LimbAutomatisms = request.form['LS_LimbAutomatisms']
                    IctalTonic_clonic = request.form['LS_IctalTonic_clonic']
                    MuscleTwitching = request.form['LS_MuscleTwitching']
                    HipThrusting = request.form['LS_HipThrusting']
                    Post_ictalFatigue = request.form['LS_Post_ictalFatigue']
                    AnyHeadInjury = request.form['LS_AnyHeadInjury']
                    PsychTraumaticEvents = request.form['LS_PsychTraumaticEvents']
                    ConcussionWithoutLOC = request.form['LS_ConcussionWithoutLOC']
                    ConcussionWithLOC = request.form['LS_ConcussionWithLOC']
                    Severe_TBILOC = request.form['LS_Severe_TBILOC']
                    Opioids = request.form['LS_Opioids']
                    SexAbuse = request.form['LS_SexAbuse']
                    PhysicalAbuse = request.form['LS_PhysicalAbuse']
                    Rape  = request.form['LS_Rape ']

                    features.append({'sex': sex, 'non_psychComorbidities': non_psychComorbidities, 'PriorAEDs': PriorAEDs,'AsthmaAttr':AsthmaAttr,
                                    'Migraine': Migraine, 'ChronicPain': ChronicPain, 'DiabetesAttr': DiabetesAttr,
                                    'non_metastaticCancer': non_metastaticCancer, 'NumberOfNoN_seizureNon_psychMedication': NumberOfNoN_seizureNon_psychMedication,
                                    'CurrentAEDs': CurrentAEDs, 'Baseline_szFreq': Baseline_szFreq,
                                    'MedianDurationOfSeizures': MedianDurationOfSeizures, 'NumberOfSeizureTypes': NumberOfSeizureTypes,
                                    'InjuryWithSeizure': InjuryWithSeizure, 'Catamenial': Catamenial, "TriggerOfSleepDeprivation": TriggerOfSleepDeprivation,
                                    'Aura': Aura, 'IctalEyeClosure': IctalEyeClosure, 'IctalHallucinations': IctalHallucinations,
                                    'OralAutomatisms': OralAutomatisms, 'Incontinence': Incontinence,
                                    'LimbAutomatisms': LimbAutomatisms, 'IctalTonic_clonic': IctalTonic_clonic,
                                    'MuscleTwitching': MuscleTwitching, 'HipThrusting': HipThrusting,
                                    'Post_ictalFatigue': Post_ictalFatigue, 'AnyHeadInjury': AnyHeadInjury,
                                    'PsychTraumaticEvents': PsychTraumaticEvents, 'ConcussionWithoutLOC': ConcussionWithoutLOC,
                                    'ConcussionWithLOC': ConcussionWithLOC,'Severe_TBILOC': Severe_TBILOC,
                                    'Opioids': Opioids, 'SexAbuse': SexAbuse,
                                    'PhysicalAbuse': PhysicalAbuse, 'Rape': Rape})

                    EpilepticSeizure = EpilepticSeizureDiagnosis()
                    results.append(EpilepticSeizure)



            
        if 'Osteoporosis' in checked:
            Joint_Pain = request.form['LS_Joint_Pain']
            sex = request.form['LS_Sex']
            age = request.form['LS_Age']
            height_in_meter = request.form['LS_height_in_meter']
            Weight_in_KG = request.form['LS_Weight_in_KG']
            Catsmoking = request.form['LS_Catsmoking'] 
            DiabetesAttr = request.form['LS_DiabetesAttr']
            Hypothyroidism = request.form['LS_Hypothyroidism']
            Seizure_Disorder = request.form['LS_Seizure_Disorder']
            Estrogen_Use = request.form['LS_Estrogen_Use'] 
            Dialysis = request.form['LS_Dialysis']
            Family_History_of_Osteo = request.form['LS_Family_History_of_Osteo']
            Maximum_Walking_distance_in_km = request.form['LS_Maximum_Walking_distance_in_km']
            Daily_Eating_habits = request.form['LS_Daily_Eating_habits']
            BMI = request.form['LS_BMI']
            Obesity = request.form['LS_Obesity']
            
            

            features.append({'Joint_Pain': Joint_Pain,'sex': sex, 'age': age,  'height_in_meter': height_in_meter,
                            'Weight_in_KG': Weight_in_KG,'Catsmoking': Catsmoking, 'DiabetesAttr': DiabetesAttr,
                            'Hypothyroidism': Hypothyroidism, 'Seizure_Disorder': Seizure_Disorder, 
                            'Estrogen_Use': Estrogen_Use, 'Dialysis': Dialysis, 
                            'Family_History_of_Osteo': Family_History_of_Osteo, 
                            'Maximum_Walking_distance_in_km': Maximum_Walking_distance_in_km,
                           'Daily_Eating_habits': Daily_Eating_habits,'BMI': BMI,  'Obesity': Obesity
                               
                                })

            osteod = Osteodiagnosis() 
            results.append(osteod)


        if 'Select All' in checked:
            sex = request.form['LS_Sex']
            age = request.form['LS_Age']
            rbc_count = request.form['LS_RBCcount']
            wbc_count = request.form['LS_WBCcount']
            bun = request.form['LS_BUN']
            creatinine = request.form['LS_Creatinine']
            hemoglobin = request.form['LS_Hemoglobin']
            hematocrit = request.form['LS_Hematocrit']
            platelet = request.form['LS_Platelet']
            mpv = request.form['LS_MPV']
            rdw = request.form['LS_RDW']
            mch = request.form['LS_MCH']
            mchc = request.form['LS_MCHC']
            mcv = request.form['LS_MCV']
            direct_bilirubin = request.form['LS_DirectBilirubin']
            total_bilirubin = request.form['LS_TotalBilirubin']
            total_protein = request.form['LS_TotalProtein']
            albumin = request.form['LS_Albumin']
            alt = request.form['LS_ALT']
            ldh = request.form['LS_LDH']
            ast = request.form['LS_AST']
            ggt = request.form['LS_GGT']
            ggtp = request.form['LS_GGTP']
            sgot = request.form['LS_SGOT']
            sgpt = request.form['LS_SGPT']
            bp_systolic = request.form['LS_BP_Systolic']
            bp_diastolic = request.form['LS_BP_Diastolic']
            respiratory_rate = request.form['LS_RespiratoryRate']
            basophil_instrument = request.form['LS_Basophil_Instrument']
            basophil_instrument_abs = request.form['LS_Basophil_Instrument_Abs']
            alk_phos = request.form['LS_Alk_Phos']
            neugran = request.form['LS_NeuGran']
            neugran_abs = request.form['LS_NeuGran_Abs']
            sodium = request.form['LS_Sodium']
            potassium = request.form['LS_Potassium']
            chloride = request.form['LS_Chloride']
            carbon_dioxide = request.form['LS_Carbon_Dioxide']
            anion_gap = request.form['LS_Anion_Gap']
            pulse = request.form['LS_Pulse']
            smoking = request.form['LS_smoking']
            sugar_level = request.form['LS_SugarLevel']
            mono = request.form['LS_MONO']
            lad = request.form['LS_LAD']
            pc_perimeter = request.form['LS_PCperimeter']
            pc_area = request.form['LS_PCarea']
            pc_smoothness = request.form['LS_PCsmoothness']
            pc_compactness = request.form['LS_PCcompactness']
            yellow_fingers = request.form['LS_yellowFingers']
            anxiety = request.form['LS_anxiety']
            wheezing = request.form['LS_wheezing']
            peer_pressure = request.form['LS_peerPressure']
            chronic_disease = request.form['LS_chronicDisease']
            fatigue = request.form['LS_fatigue']
            allergy = request.form['LS_allergy']
            coughing = request.form['LS_coughing']
            alcohol = request.form['LS_alcohol']
            shortness_of_breath = request.form['LS_shortness_of_Breath']
            swallowing_difficulty = request.form['LS_swallowing_Difficulty']
            chest_pain = request.form['LS_chest_Pain']
            at = request.form['LS_at']
            ean = request.form['LS_ean']
            mhci = request.form['LS_mhci']
            vasi = request.form['LS_vasi']
            varg = request.form['LS_varg']
            vars = request.form['LS_vars']
            tmi = request.form['LS_tmi']
            ndays = request.form['LS_NDays']
            hepatomegaly = request.form['LS_Hepatomegaly']
            spiders = request.form['LS_Spiders']
            edema = request.form['LS_Edema']
            cholesterol = request.form['LS_Cholesterol']
            copper = request.form['LS_Copper']
            prothrombin = request.form['LS_Prothrombin']
            ascites = request.form['LS_Ascites']
            bilirubin = request.form['LS_SerumBilirubin']
            triglycerides = request.form['LS_Triglycerides']
            drug = request.form['LS_Drug']
            status_lc = request.form['LS_StatusLC']
            wbc = request.form['LS_WBC']
            rbc = request.form['LS_RBC']
            num_of_diag = request.form['LS_NumOfDiag']
            condylomatosis = request.form['LS_Condylomatosis']
            std_syphilis = request.form['LS_StdSyphilis']
            hiv = request.form['LS_HIV']
            std_hpv = request.form['LS_STD_HPV']
            dx = request.form['LS_Dx']
            dxcin = request.form['LS_DxCIN']
            dxhpv = request.form['LS_DxHPV']
            lymphocyte = request.form['LS_Lymphocyte']
            hdl = request.form['LS_HDL']
            fall_risk = request.form['LS_FallRiskMorse']
            weight = request.form['LS_Weight']
            body_mass = request.form['LS_BodyMass']
            int_norm_rati = request.form['LS_IntNormRati']
            hh_size = request.form['LS_hhsize']
            edu_level = request.form['LS_EducationLevel']
            val_livestock = request.form['LS_ValOfLivestock']
            val_durab_good = request.form['LS_ValOfDurabGood']
            val_saving = request.form['LS_ValSaving']
            land_owned = request.form['LS_LandOwned']
            alcohol_consumed = request.form['LS_AlcoholConsumed']
            tobacco_consumed = request.form['LS_TobaccoConsumed']
            edu_expenditure = request.form['LS_EduExpenditure']
            ent_nonag_flowcost = request.form['LS_Ent_nonag_flowcost']
            ent_animalstockrev = request.form['LS_Ent_animalstockrev']
            ent_total_cost = request.form['LS_Ent_total_cost']
            fs_adwholed_often = request.form['LS_Fs_adwholed_often']
            nondurable_investment = request.form['LS_Nondurable_investment']
            amount_received_mpesa = request.form['LS_Amount_received_mpesa']
            married = request.form['LS_Married']
            children = request.form['LS_Children']
            hh_children = request.form['LS_hh_children']
            ent_nonagbusiness = request.form['LS_Ent_nonagbusiness']
            saved_mpesa = request.form['LS_Saved_mpesa']
            early_survey = request.form['LS_Early_survey']
            imagery_part_min = request.form['LS_Imagery_part_min']
            imagery_part_avg = request.form['LS_Imagery_part_avg']
            real_part_min = request.form['LS_Real_part_min']
            real_part_avg = request.form['LS_Real_part_avg'],
            request.form['LS_PCV']
            # tribe = request.form['LS_Tribe']
            Joint_Pain = request.form['LS_Joint_Pain']
            height_in_meter = request.form['LS_height_in_meter']
            Weight_in_KG = request.form['LS_Weight_in_KG']
            # Alcoholic = request.form['LS_Alcoholic']
            Hypothyroidism = request.form['LS_Hypothyroidism']
            Seizure_Disorder = request.form['LS_Seizure_Disorder']
            Estrogen_Use = request.form['LS_Estrogen_Use']
            # History_of_Fracture = request.form['LS_History_of_Fracture']
            Dialysis = request.form['LS_Dialysis']
            Family_History_of_Osteo = request.form['LS_Family_History_of_Osteo']
            Maximum_Walking_distance_in_km = request.form['LS_Maximum_Walking_distance_in_km']
            Daily_Eating_habits = request.form['LS_Daily_Eating_habits']
            BMI = request.form['LS_BMI']
            # Site = request.form['LS_Site']
            Obesity = request.form['LS_Obesity']
            Catsmoking = request.form['LS_Catsmoking']
            non_psychComorbidities = request.form['LS_non_psychComorbidities']
            PriorAEDs = request.form['LS_PriorAEDs']
            AsthmaAttr = request.form['LS_AsthmaAttr']
            Migraine = request.form['LS_Migraine']
            ChronicPain = request.form['LS_ChronicPain']
            DiabetesAttr = request.form['LS_DiabetesAttr']
            non_metastaticCancer = request.form['LS_non_metastaticCancer']
            NumberOfNoN_seizureNon_psychMedication = request.form['LS_NumberOfNoN_seizureNon_psychMedication']
            CurrentAEDs = request.form['LS_CurrentAEDs']
            Baseline_szFreq = request.form['LS_Baseline_szFreq']
            MedianDurationOfSeizures = request.form['LS_MedianDurationOfSeizures']
            NumberOfSeizureTypes = request.form['LS_NumberOfSeizureTypes']
            InjuryWithSeizure = request.form['LS_InjuryWithSeizure']
            Catamenial = request.form['LS_Catamenial']
            TriggerOfSleepDeprivation = request.form['LS_TriggerOfSleepDeprivation']
            Aura = request.form['LS_Aura']
            IctalEyeClosure = request.form['LS_IctalEyeClosure']
            IctalHallucinations = request.form['LS_IctalHallucinations']
            OralAutomatisms = request.form['LS_OralAutomatisms']
            Incontinence  = request.form['LS_Incontinence']
            LimbAutomatisms = request.form['LS_LimbAutomatisms']
            IctalTonic_clonic = request.form['LS_IctalTonic_clonic']
            MuscleTwitching = request.form['LS_MuscleTwitching']
            HipThrusting = request.form['LS_HipThrusting']
            Post_ictalFatigue = request.form['LS_Post_ictalFatigue']
            AnyHeadInjury = request.form['LS_AnyHeadInjury']
            PsychTraumaticEvents = request.form['LS_PsychTraumaticEvents']
            ConcussionWithoutLOC = request.form['LS_ConcussionWithoutLOC']
            ConcussionWithLOC = request.form['LS_ConcussionWithLOC']
            Severe_TBILOC = request.form['LS_Severe_TBILOC']
            Opioids = request.form['LS_Opioids']
            SexAbuse = request.form['LS_SexAbuse']
            PhysicalAbuse = request.form['LS_PhysicalAbuse']
            Rape  = request.form['LS_Rape']
            protein= request.form['LS_ProtienUA']
            glucose = request.form['LS_Glucose']
            hemoglobin = request.form['LS_Hemoglobin']
            hdl= request.form['LS_HDL']
            ldl = request.form['LS_HDL_LDL']
            cholesterol = request.form['LS_Cholesterol']
            ldl_chol = request.form['LS_LDL_Cholesterol']
            pH= request.form['LS_pH_Urine']
            calcium = request.form['LS_Calcium']
            smoke= request.form['LS_smoking']
            pesticide = request.form['LS_pesticide']
            skin_cancer_history = request.form['LS_skin_cancer_hist']
            cancer_history	= request.form['LS_cancer_hist']
            has_piped_water = request.form['LS_has_piped_water']
            has_sewage_system = request.form['LS_has_sewage_system']
            p16540= request.form['LS_p16540']
            p16580 = request.form['LS_p16580']
            mdm2 = request.form['LS_mdm2']
            GAL3 = request.form['LS_GAL3']
            TIM1 = request.form['LS_TIM1']
            # BASEM TEAM : please add the features you need



            features.append({
                        'sex': sex,
                        'age': age,
                        'rbc_count': rbc_count,
                        'wbc_count': wbc_count,
                        'bun': bun,
                        'creatinine': creatinine,
                        'hemoglobin': hemoglobin,
                        'hematocrit': hematocrit,
                        'platelet': platelet,
                        'mpv': mpv,
                        'rdw': rdw,
                        'mch': mch,
                        'mchc': mchc,
                        'mcv': mcv,
                        'direct_bilirubin': direct_bilirubin,
                        'total_bilirubin': total_bilirubin,
                        'total_protein': total_protein,
                        'albumin': albumin,
                        'alt': alt,
                        'ldh': ldh,
                        'ast': ast,
                        'ggt': ggt,
                        'ggtp': ggtp,
                        'sgot': sgot,
                        'sgpt': sgpt,
                        'bp_systolic': bp_systolic,
                        'bp_diastolic': bp_diastolic,
                        'respiratory_rate': respiratory_rate,
                        'basophil_instrument': basophil_instrument,
                        'basophil_instrument_abs': basophil_instrument_abs,
                        'alk_phos': alk_phos,
                        'neugran': neugran,
                        'neugran_abs': neugran_abs,
                        'sodium': sodium,
                        'potassium': potassium,
                        'chloride': chloride,
                        'carbon_dioxide': carbon_dioxide,
                        'anion_gap': anion_gap,
                        'pulse': pulse,
                        'smoking': smoking,
                        'sugar_level': sugar_level,
                        'mono': mono,
                        'lad': lad,
                        'pc_perimeter': pc_perimeter,
                        'pc_area': pc_area,
                        'pc_smoothness': pc_smoothness,
                        'pc_compactness': pc_compactness,
                        'yellow_fingers': yellow_fingers,
                        'anxiety': anxiety,
                        'wheezing': wheezing,
                        'peer_pressure': peer_pressure,
                        'chronic_disease': chronic_disease,
                        'fatigue': fatigue,
                        'allergy': allergy,
                        'coughing': coughing,
                        'alcohol': alcohol,
                        'shortness_of_breath': shortness_of_breath,
                        'swallowing_difficulty': swallowing_difficulty,
                        'chest_pain': chest_pain,
                        'at': at,
                        'ean': ean,
                        'mhci': mhci,
                        'vasi': vasi,
                        'varg': varg,
                        'vars': vars,
                        'tmi': tmi,
                        'ndays': ndays,
                        'hepatomegaly': hepatomegaly,
                        'spiders': spiders,
                        'edema': edema,
                        'cholesterol': cholesterol,
                        'copper': copper,
                        'prothrombin': prothrombin,
                        'ascites': ascites,
                        'bilirubin': bilirubin,
                        'triglycerides': triglycerides,
                        'drug': drug,
                        'status_lc': status_lc,
                        'wbc': wbc,
                        'rbc': rbc,
                        'num_of_diag': num_of_diag,
                        'condylomatosis': condylomatosis,
                        'std_syphilis': std_syphilis,
                        'hiv': hiv,
                        'std_hpv': std_hpv,
                        'dx': dx,
                        'dxcin': dxcin,
                        'dxhpv': dxhpv,
                        'lymphocyte': lymphocyte,
                        'hdl': hdl,
                        'fall_risk': fall_risk,
                        'weight': weight,
                        'body_mass': body_mass,
                        'int_norm_rati': int_norm_rati,
                        'hh_size': hh_size,
                        'edu_level': edu_level,
                        'val_livestock': val_livestock,
                        'val_durab_good': val_durab_good,
                        'val_saving': val_saving,
                        'land_owned': land_owned,
                        'alcohol_consumed': alcohol_consumed,
                        'tobacco_consumed': tobacco_consumed,
                        'edu_expenditure': edu_expenditure,
                        'ent_nonag_flowcost': ent_nonag_flowcost,
                        'ent_animalstockrev': ent_animalstockrev,
                        'ent_total_cost': ent_total_cost,
                        'fs_adwholed_often': fs_adwholed_often,
                        'nondurable_investment': nondurable_investment,
                        'amount_received_mpesa': amount_received_mpesa,
                        'married': married,
                        'children': children,
                        'hh_children': hh_children,
                        'ent_nonagbusiness': ent_nonagbusiness,
                        'saved_mpesa': saved_mpesa,
                        'early_survey': early_survey,
                        'imagery_part_min': imagery_part_min,
                        'imagery_part_avg': imagery_part_avg,
                        'real_part_min': real_part_min,
                        'real_part_avg': real_part_avg,
                        'Joint_Pain': Joint_Pain, 
                        'height_in_meter': height_in_meter,
                        'Weight_in_KG': Weight_in_KG, 
                        # 'Alcoholic': Alcoholic,
                        'Hypothyroidism': Hypothyroidism, 
                        'Seizure_Disorder': Seizure_Disorder,
                        'Estrogen_Use': Estrogen_Use, 
                        # 'History_of_Fracture': History_of_Fracture,
                        'Dialysis': Dialysis,
                        'Family_History_of_Osteo': Family_History_of_Osteo, 
                        'Maximum_Walking_distance_in_km': Maximum_Walking_distance_in_km,
                        'Daily_Eating_habits': Daily_Eating_habits, 
                        'BMI': BMI, 
                        # 'Site': Site,
                        'Obesity': Obesity,
                        'Catsmoking': Catsmoking,
                        # 'tribe':tribe,
                        'non_psychComorbidities': non_psychComorbidities, 
                        'PriorAEDs': PriorAEDs,
                        'Migraine': Migraine, 
                        'ChronicPain': ChronicPain, 
                        'non_metastaticCancer': non_metastaticCancer, 
                        'NumberOfNoN_seizureNon_psychMedication': NumberOfNoN_seizureNon_psychMedication,
                        'CurrentAEDs': CurrentAEDs, 'Baseline_szFreq': Baseline_szFreq,
                        'MedianDurationOfSeizures': MedianDurationOfSeizures, 
                        'NumberOfSeizureTypes': NumberOfSeizureTypes,
                        'InjuryWithSeizure': InjuryWithSeizure, 'Catamenial': Catamenial,
                        'Aura': Aura, 'IctalEyeClosure': IctalEyeClosure, 
                        'IctalHallucinations': IctalHallucinations,
                        'OralAutomatisms': OralAutomatisms, 'Incontinence': Incontinence,
                        'LimbAutomatisms': LimbAutomatisms, 'IctalTonic_clonic': IctalTonic_clonic,
                        'MuscleTwitching': MuscleTwitching, 'HipThrusting': HipThrusting,
                        'Post_ictalFatigue': Post_ictalFatigue, 'AnyHeadInjury': AnyHeadInjury,
                        'PsychTraumaticEvents': PsychTraumaticEvents, 'ConcussionWithoutLOC': ConcussionWithoutLOC,
                        'ConcussionWithLOC': ConcussionWithLOC,'Severe_TBILOC': Severe_TBILOC,
                        'Opioids': Opioids, 'SexAbuse': SexAbuse,
                        'PhysicalAbuse': PhysicalAbuse, 'Rape': Rape,'AsthmaAttr':AsthmaAttr, 'DiabetesAttr': DiabetesAttr,
                        'protein': protein, 'glucose': glucose, 'hemoglobin': hemoglobin,
                             'hdl': hdl, 'ldl': ldl, 'cholesterol': cholesterol, 'ldl_chol': ldl_chol,
                             'pH': pH, 'calcium': calcium , 'smoke': smoke, 'pesticide': pesticide,
                                'skin_cancer_history': skin_cancer_history, 'cancer_history': cancer_history,
                                'has_piped_water': has_piped_water, 'has_sewage_system': has_sewage_system,
                                'p16540': p16540, 'p16580': p16580, 'mdm2': mdm2, 'GAL3': GAL3, 'TIM1': TIM1
                                # BASEM TEAM : please add the features you need
                    })

            diagnoses = [ckddiagnosis(), dmdiagnosis(), CHDDdiagnosis(), RAdiagnosis(), asdiagnosis(), tcdiagnosis(),
                 Schizodiagnosis(), Hypodiagnosis(), PCdiagnosis(), MSdiagnosis(), ADdiagnosis(), LCdiagnosis(),
                 Glaucomadiagnosis(), LCHdiagnosis(), PRDdiagnosis(), CervicalCancerdiagnosis(), HepCdiagnosis(),
                 Depressiondiagnosis(), COPDdiagnosis(), Hyperdiagnosis(), SkinDiagnosis(),ColorectDiagnosis(), SCAdiagnosis(), EpilepticSeizureDiagnosis(), Osteodiagnosis(),
                 pc2diagnosis(), strokediagnosis(), pcosdiagnosis() ]

            for diagnosis in diagnoses:
                results.append(diagnosis)


        features = {key: value for dict_ in features for key, value in dict_.items()}

        db, client = connection()

        mrn = request.form['LS_PatientMRN']

        if mrn == '':
            return render_template("LSdiagnosis.html", mrn='False')

        else:

            existence =db.patient.find_one({"_id": int(mrn)})

            if existence is None:
                return render_template("LSdiagnosis.html", patient='False')

            if len(checked)==0:
                return render_template("LSdiagnosis.html", checkedlist='False')

            else:
                tests_count = db.test.count_documents({})

                test = {
                    'mrn': int(mrn),
                    'test_id': tests_count + 1,
                    'predicted_disease': results,
                    'test_date': str(date.today())
                }

                test.update(features)
                db.test.insert_one(test)
                return render_template("LSdiagnosis.html", patient='True')

        return render_template("LSdiagnosis.html" )

@app.route('/CKDdiagnosis', methods=['POST'])
def ckddiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Chronic Kidney Disease", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # **check Path**

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                crt = request.form['LS_Creatinine']
                bun = request.form['LS_BUN']

                prd = loadedmodel.predict([[float(bun), float(crt)]])

                if prd[0] == "CKD":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Chronic Kidney Disease',
                    'prediction': result,
                    'accuracy': accuracy
                }

                return prediction
            else:
                crt = request.form['crt']
                bun = request.form['bun']
                prd = loadedmodel.predict([[float(bun), float(crt)]])

                if prd[0] == "CKD":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Chronic Kidney Disease"

                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            crt = request.form['crt']
            bun = request.form['bun']
            prd = loadedmodel.predict([[float(bun), float(crt)]])

            if prd[0] == "CKD":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Chronic Kidney Disease"

            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/DMdiagnosis', methods=['POST'])
def dmdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Diabetes", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)
        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # **check Path**

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                slg = request.form['LS_SugarLevel']
                hct = request.form['LS_Hematocrit']
                mpv = request.form['LS_MPV']

                prd = loadedmodel.predict([[float(slg), float(hct), float(mpv)]])

                if prd[0] == "DM":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Diabetes Mellitus',
                    'prediction': result,
                    'accuracy': accuracy
                }

                return prediction
            else:
                slg = request.form['slg']
                hct = request.form['hct']
                mpv = request.form['mpv']

                prd = loadedmodel.predict([[float(slg), float(hct), float(mpv)]])


                if prd[0] == "DM":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Diabetes Mellitus"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            slg = request.form['slg']
            hct = request.form['hct']
            mpv = request.form['mpv']

            prd = loadedmodel.predict([[float(slg), float(hct), float(mpv)]])

            print(prd)

            if prd[0] == "DM":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Diabetes Mellitus"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/CHDDdiagnosis', methods=['POST'])
def CHDDdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Coronary Heart Disease", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        pathForModel = open('./static/model/' + modelName, 'rb')
        loadedmodel = pickle.load(pathForModel)

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                gender = request.form['LS_Sex']
                age = request.form['LS_Age']
                rdw = request.form['LS_RDW']
                platelet = request.form['LS_Platelet']
                mpv = request.form['LS_MPV']
                mono = request.form['LS_MONO']
                basophil_instrument = request.form['LS_Basophil_Instrument'] #%
                basophil_instrument_abs = request.form['LS_Basophil_Instrument_Abs'] ##
                potassium = request.form['LS_Potassium']
                ggtp = request.form['LS_GGTP']
                sgpt = request.form['LS_SGPT']
                sgot = request.form['LS_SGOT']
                neugran = request.form['LS_NeuGran']
                neugran_abs = request.form['LS_NeuGran_Abs']
                anion_gap = request.form['LS_Anion_Gap']

                if gender == "Male":
                    gender = 0
                else:
                    gender = 1

                prd = loadedmodel.predict([[float(gender), float(age), float(rdw), float(platelet),
                                            float(mpv), float(neugran), float(basophil_instrument),
                                            float(basophil_instrument_abs), float(neugran_abs), float(mono),
                                            float(potassium), float(anion_gap), float(ggtp), float(sgot), float(sgpt)]])
                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Coronary Heart Disease',
                    'prediction': result,
                    'accuracy': accuracy
                }

                return prediction
            else:
                gender = request.form['genderCHDD']
                age = request.form['AgeCHDD']
                rdw = request.form['RDWCHDD']
                platelet = request.form['Platelet_CountCHDD']
                mpv = request.form['MVPCHDD']
                mono = request.form['MONOCHDD']
                basophil_instrument = request.form['BasopCHDD']
                basophil_instrument_abs = request.form['BASOCHDD']
                potassium = request.form['POTASSIUMCHDD']
                ggtp = request.form['GGTPCHDD']
                sgpt = request.form['SGOTCHDD']
                sgot = request.form['SGPTCHDD']
                neugran = request.form['NGICHDD']
                neugran_abs = request.form['NGIACHDD']
                anion_gap = request.form['AGCHDD']

                if gender == "Male":
                    gender = 0
                else:
                    gender = 1

                prd = loadedmodel.predict([[float(gender), float(age), float(rdw), float(platelet),
                                            float(mpv), float(neugran), float(basophil_instrument),
                                            float(basophil_instrument_abs), float(neugran_abs), float(mono),
                                            float(potassium), float(anion_gap), float(ggtp), float(sgot), float(sgpt)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Coronary Heart Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            gender = request.form['genderCHDD']
            age = request.form['AgeCHDD']
            rdw = request.form['RDWCHDD']
            platelet = request.form['Platelet_CountCHDD']
            mpv = request.form['MVPCHDD']
            mono = request.form['MONOCHDD']
            basophil_instrument = request.form['BasopCHDD']
            basophil_instrument_abs = request.form['BASOCHDD']
            potassium = request.form['POTASSIUMCHDD']
            ggtp = request.form['GGTPCHDD']
            sgpt = request.form['SGOTCHDD']
            sgot = request.form['SGPTCHDD']
            neugran = request.form['NGICHDD']
            neugran_abs = request.form['NGIACHDD']
            anion_gap = request.form['AGCHDD']

            if gender == "Male":
                gender = 0
            else:
                gender = 1

            prd = loadedmodel.predict([[float(gender), float(age), float(rdw), float(platelet),
                                        float(mpv), float(neugran), float(basophil_instrument),
                                        float(basophil_instrument_abs), float(neugran_abs), float(mono),
                                        float(potassium), float(anion_gap), float(ggtp), float(sgot), float(sgpt)]])

            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Coronary Heart Disease"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/RAdiagnosis', methods=['POST'])
def RAdiagnosis():
    try:

        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Rheumatoid Arthritis", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                genderRA = request.form['LS_Sex']
                ageRA = request.form['LS_Age']
                albuminRA = request.form['LS_Albumin']
                AlkPhosRA = request.form['LS_Alk_Phos']
                bunRA = request.form['LS_BUN']
                ClRA = request.form['LS_Chloride']
                CO2RA = request.form['LS_Carbon_Dioxide']
                creatRA = request.form['LS_Creatinine']
                dbiliRA = request.form['LS_DirectBilirubin']
                GGTPRA = request.form['LS_GGTP']
                hgbRA = request.form['LS_Hemoglobin']
                hctRA = request.form['LS_Hematocrit']
                KRA = request.form['LS_Potassium']
                LDHRA = request.form['LS_LAD']
                MCHRA = request.form['LS_MCH']
                MCHCRA = request.form['LS_MCHC']
                MCVRA = request.form['LS_MCV']
                MPVRA = request.form['LS_MPV']
                NaRA = request.form['LS_Sodium']
                PltRA = request.form['LS_Platelet']
                RBCRA = request.form['LS_RBCcount']
                RDWRA = request.form['LS_RDW']
                SGOTRA = request.form['LS_SGOT']
                SGPTRA = request.form['LS_SGPT']
                TbiliRA = request.form['LS_TotalBilirubin']
                TProteinRA = request.form['LS_TotalProtein']

                if genderRA == "Male":
                    genderRA = 0
                else:
                    genderRA = 1

                prd = loadedmodel.predict([[float(genderRA), float(ageRA), float(albuminRA), float(AlkPhosRA),
                                            float(bunRA), float(ClRA), float(CO2RA), float(creatRA), float(dbiliRA),
                                            float(GGTPRA), float(hgbRA), float(hctRA), float(KRA), float(LDHRA),
                                            float(MCHRA), float(MCHCRA), float(MCVRA), float(MPVRA), float(NaRA),
                                            float(PltRA), float(RBCRA), float(RDWRA), float(SGOTRA), float(SGPTRA),
                                            float(TbiliRA), float(TProteinRA)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Rheumatoid Arthritis',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                genderRA = request.form['genderRA']
                ageRA = request.form['ageRA']
                albuminRA = request.form['albuminRA']
                AlkPhosRA = request.form['AlkPhosRA']
                bunRA = request.form['bunRA']
                ClRA = request.form['ClRA']
                CO2RA = request.form['CO2RA']
                creatRA = request.form['creatRA']
                dbiliRA = request.form['dbiliRA']
                GGTPRA = request.form['GGTPRA']
                hgbRA = request.form['hgbRA']
                hctRA = request.form['hctRA']
                KRA = request.form['KRA']
                LDHRA = request.form['LDHRA']
                MCHRA = request.form['MCHRA']
                MCHCRA = request.form['MCHCRA']
                MCVRA = request.form['MCVRA']
                MPVRA = request.form['MPVRA']
                NaRA = request.form['NaRA']
                PltRA = request.form['PltRA']
                RBCRA = request.form['RBCRA']
                RDWRA = request.form['RDWRA']
                SGOTRA = request.form['SGOTRA']
                SGPTRA = request.form['SGPTRA']
                TbiliRA = request.form['TbiliRA']
                TProteinRA = request.form['TProteinRA']

                if genderRA == "Male":
                    genderRA = 0
                else:
                    genderRA = 1

                prd = loadedmodel.predict([[float(genderRA), float(ageRA), float(albuminRA), float(AlkPhosRA),
                                            float(bunRA), float(ClRA), float(CO2RA), float(creatRA), float(dbiliRA),
                                            float(GGTPRA), float(hgbRA), float(hctRA), float(KRA), float(LDHRA),
                                            float(MCHRA), float(MCHCRA), float(MCVRA), float(MPVRA), float(NaRA),
                                            float(PltRA), float(RBCRA), float(RDWRA), float(SGOTRA), float(SGPTRA),
                                            float(TbiliRA), float(TProteinRA)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Rheumatoid Arthritis"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            genderRA = request.form['genderRA']
            ageRA = request.form['ageRA']
            albuminRA = request.form['albuminRA']
            AlkPhosRA = request.form['AlkPhosRA']
            bunRA = request.form['bunRA']
            ClRA = request.form['ClRA']
            CO2RA = request.form['CO2RA']
            creatRA = request.form['creatRA']
            dbiliRA = request.form['dbiliRA']
            GGTPRA = request.form['GGTPRA']
            hgbRA = request.form['hgbRA']
            hctRA = request.form['hctRA']
            KRA = request.form['KRA']
            LDHRA = request.form['LDHRA']
            MCHRA = request.form['MCHRA']
            MCHCRA = request.form['MCHCRA']
            MCVRA = request.form['MCVRA']
            MPVRA = request.form['MPVRA']
            NaRA = request.form['NaRA']
            PltRA = request.form['PltRA']
            RBCRA = request.form['RBCRA']
            RDWRA = request.form['RDWRA']
            SGOTRA = request.form['SGOTRA']
            SGPTRA = request.form['SGPTRA']
            TbiliRA = request.form['TbiliRA']
            TProteinRA = request.form['TProteinRA']

            if genderRA == "Male":
                genderRA = 0
            else:
                genderRA = 1

            prd = loadedmodel.predict([[float(genderRA), float(ageRA), float(albuminRA), float(AlkPhosRA),
                                        float(bunRA), float(ClRA), float(CO2RA), float(creatRA), float(dbiliRA),
                                        float(GGTPRA), float(hgbRA), float(hctRA), float(KRA), float(LDHRA),
                                        float(MCHRA), float(MCHCRA), float(MCVRA), float(MPVRA), float(NaRA),
                                        float(PltRA), float(RBCRA), float(RDWRA), float(SGOTRA), float(SGPTRA),
                                        float(TbiliRA), float(TProteinRA)]])

            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Rheumatoid Arthritis"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
    finally:
        client.close()

@app.route('/Schizodiagnosis', methods=['POST'])
def Schizodiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Schizophrenia", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                age = request.form['LS_Age']
                WBCcount = request.form['LS_WBCcount']
                Hemoglobin = request.form['LS_Hemoglobin']
                Hematocrit = request.form['LS_Hematocrit']
                MCV = request.form['LS_MCV']
                MCH = request.form['LS_MCH']
                MCHC = request.form['LS_MCHC']
                Platelet = request.form['LS_Platelet']
                MPV = request.form['LS_MPV']
                AST = request.form['LS_AST']
                TotalProtein = request.form['LS_TotalProtein']
                GGT = request.form['LS_GGT']

                prd = loadedmodel.predict([[float(age), float(WBCcount),float(Hemoglobin), float(Hematocrit),
                                            float(MCV), float(MCH),float(MCHC), float(Platelet),
                                            float(MPV), float(AST),float(TotalProtein), float(GGT) ]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Schizophrenia',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                age = request.form['AgeSchizo']
                WBCcount = request.form['WBCcSchizo']
                Hemoglobin = request.form['HGBSchizo']
                Hematocrit = request.form['HCTSchizo']
                MCV = request.form['MCVSchizo']
                MCH = request.form['MCHSchizo']
                MCHC = request.form['MCHCSchizo']
                Platelet = request.form['PLTSchizo']
                MPV = request.form['MPVSchizo']
                AST = request.form['ASTSchizo']
                TotalProtein = request.form['TotProtSchizo']
                GGT = request.form['GGTSchizo']


                prd = loadedmodel.predict([[float(age), float(WBCcount), float(Hemoglobin), float(Hematocrit),
                                            float(MCV), float(MCH), float(MCHC), float(Platelet),
                                            float(MPV), float(AST), float(TotalProtein), float(GGT)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Schizophrenia"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            age = request.form['AgeSchizo']
            WBCcount = request.form['WBCcSchizo']
            Hemoglobin = request.form['HGBSchizo']
            Hematocrit = request.form['HCTSchizo']
            MCV = request.form['MCVSchizo']
            MCH = request.form['MCHSchizo']
            MCHC = request.form['MCHCSchizo']
            Platelet = request.form['PLTSchizo']
            MPV = request.form['MPVSchizo']
            AST = request.form['ASTSchizo']
            TotalProtein = request.form['TotProtSchizo']
            GGT = request.form['GGTSchizo']

            prd = loadedmodel.predict([[float(age), float(WBCcount), float(Hemoglobin), float(Hematocrit),
                                        float(MCV), float(MCH), float(MCHC), float(Platelet),
                                        float(MPV), float(AST), float(TotalProtein), float(GGT)]])

            if prd[0] == 1:
                result = "Positive"
            else:
                result = "Negative"
            disease = "Schizophrenia"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

# The new Thyroid Cancer Diagnosis
@app.route('/tcdiagnosis', methods=['POST'])
def tcdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Thyroid Cancer", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                genderThyroid = request.form['LS_Sex']
                ageThyroid = request.form['LS_Age']
                HematocritThyroid = request.form['LS_Hematocrit']
                MCHCThyroid = request.form['LS_MCHC']
                MPVThyroid = request.form['LS_MPV']
                RBCThyroid = request.form['LS_RBCcount']
                WBCThyroid = request.form['LS_WBCcount']

                if genderThyroid == "Male":
                    genderThyroid = 0
                else:
                    genderThyroid = 1

                prd = loadedmodel.predict([[float(genderThyroid), float(ageThyroid), float(HematocritThyroid),
                                            float(MCHCThyroid), float(MPVThyroid), float(RBCThyroid),
                                            float(WBCThyroid)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Thyroid Cancer',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                genderThyroid = request.form['genderThyroid']
                ageThyroid = request.form['ageThyroid']
                HematocritThyroid = request.form['HematocritThyroid']
                MCHCThyroid = request.form['MCHCThyroid']
                MPVThyroid = request.form['MPVThyroid']
                RBCThyroid = request.form['RBCThyroid']
                WBCThyroid = request.form['WBCThyroid']

                if genderThyroid == "Male":
                    genderThyroid = 0
                else:
                    genderThyroid = 1

                prd = loadedmodel.predict([[float(genderThyroid), float(ageThyroid), float(HematocritThyroid),
                                            float(MCHCThyroid), float(MPVThyroid), float(RBCThyroid), float(WBCThyroid)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Thyroid Cancer"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            genderThyroid = request.form['genderThyroid']
            ageThyroid = request.form['ageThyroid']
            HematocritThyroid = request.form['HematocritThyroid']
            MCHCThyroid = request.form['MCHCThyroid']
            MPVThyroid = request.form['MPVThyroid']
            RBCThyroid = request.form['RBCThyroid']
            WBCThyroid = request.form['WBCThyroid']

            if genderThyroid == "Male":
                genderThyroid = 0
            else:
                genderThyroid = 1

            prd = loadedmodel.predict([[float(genderThyroid), float(ageThyroid), float(HematocritThyroid),
                                        float(MCHCThyroid), float(MPVThyroid), float(RBCThyroid), float(WBCThyroid)]])

            if prd[0] == 1:
                result = "Positive"
            else:
                result = "Negative"
            disease = "Thyroid Cancer"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

# The new Asthma  Diagnosis
@app.route('/asdiagnosis', methods=['POST'])
def asdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Asthma", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                genderAsthma = request.form['LS_Sex']
                ageAsthma = request.form['LS_Age']
                BasophilsAsthma = request.form['LS_Basophil_Instrument']
                HematocritAsthma = request.form['LS_Hematocrit']
                HemoglobinAsthma = request.form['LS_Hemoglobin']
                MCHAsthma = request.form['LS_MCH']
                MCHCAsthma = request.form['LS_MCHC']
                MPVAsthma = request.form['LS_MPV']
                WBCAsthma = request.form['LS_WBCcount']

                if genderAsthma == "Male":
                    genderAsthma = 0
                else:
                    genderAsthma = 1

                prd = loadedmodel.predict([[float(genderAsthma), float(ageAsthma), float(BasophilsAsthma),
                                            float(HematocritAsthma), float(HemoglobinAsthma), float(MCHAsthma),
                                            float(MCHCAsthma), float(MPVAsthma), float(WBCAsthma)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Asthma Disease',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                genderAsthma = request.form['genderAsthma']
                ageAsthma = request.form['ageAsthma']
                BasophilsAsthma = request.form['BasophilsAsthma']
                HematocritAsthma = request.form['HematocritAsthma']
                HemoglobinAsthma = request.form['HemoglobinAsthma']
                MCHAsthma = request.form['MCHAsthma']
                MCHCAsthma = request.form['MCHCAsthma']
                MPVAsthma = request.form['MPVAsthma']
                WBCAsthma = request.form['WBCAsthma']

                if genderAsthma == "Male":
                    genderAsthma = 0
                else:
                    genderAsthma = 1

                prd = loadedmodel.predict([[float(genderAsthma), float(ageAsthma), float(BasophilsAsthma),
                                            float(HematocritAsthma), float(HemoglobinAsthma), float(MCHAsthma),
                                            float(MCHCAsthma), float(MPVAsthma), float(WBCAsthma)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Asthma Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            genderAsthma = request.form['genderAsthma']
            ageAsthma = request.form['ageAsthma']
            BasophilsAsthma = request.form['BasophilsAsthma']
            HematocritAsthma = request.form['HematocritAsthma']
            HemoglobinAsthma = request.form['HemoglobinAsthma']
            MCHAsthma = request.form['MCHAsthma']
            MCHCAsthma = request.form['MCHCAsthma']
            MPVAsthma = request.form['MPVAsthma']
            WBCAsthma = request.form['WBCAsthma']

            if genderAsthma == "Male":
                genderAsthma = 0
            else:
                genderAsthma = 1

            prd = loadedmodel.predict([[float(genderAsthma), float(ageAsthma), float(BasophilsAsthma),
                                        float(HematocritAsthma), float(HemoglobinAsthma), float(MCHAsthma),
                                        float(MCHCAsthma), float(MPVAsthma), float(WBCAsthma)]])

            if prd[0] == 1:
                result = "Positive"
            else:
                result = "Negative"
            disease = "Asthma Disease"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

# OUR MODELS 2021
@app.route('/Hypodiagnosis', methods=['POST'])
def Hypodiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Hypothyroidism", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                ageHypo = request.form['LS_Age']
                BPSHypo = request.form['LS_BP_Systolic']
                RespHypo = request.form['LS_RespiratoryRate']
                MCVHypo = request.form['LS_MCV']
                POHypo = request.form['LS_Pulse']

                prd = loadedmodel.predict(
                    [[float(ageHypo), float(BPSHypo), float(RespHypo), float(MCVHypo), float(POHypo)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Hypothyroidism Disease',
                    'prediction': result,
                    'accuracy': accuracy
                }

                return prediction
            else:
                ageHypo = request.form['ageHypo']
                BPSHypo = request.form['BPSHypo']
                RespHypo = request.form['RespHypo']
                MCVHypo = request.form['MCVHypo']
                POHypo = request.form['POHypo']

                prd = loadedmodel.predict(
                    [[float(ageHypo), float(BPSHypo), float(RespHypo), float(MCVHypo), float(POHypo)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"
                    disease = "Hypothyroidism Disease"
                    return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            ageHypo = request.form['ageHypo']
            BPSHypo = request.form['BPSHypo']
            RespHypo = request.form['RespHypo']
            MCVHypo = request.form['MCVHypo']
            POHypo = request.form['POHypo']

            prd = loadedmodel.predict(
                [[float(ageHypo), float(BPSHypo), float(RespHypo), float(MCVHypo), float(POHypo)]])

            if prd[0] == 1:
                result = "Positive"
            else:
                result = "Negative"
                disease = "Hypothyroidism Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/PCdiagnosis', methods=['POST'])
def PCdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Prostate Cancer", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                PCperimeter = request.form['LS_PCperimeter']
                PCarea = request.form['LS_PCarea']
                PCsmoothness = request.form['LS_PCsmoothness']
                PCcompactness = request.form['LS_PCcompactness']

                prd = loadedmodel.predict(
                    [[float(PCperimeter), float(PCarea), float(PCsmoothness), float(PCcompactness)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'PC',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                PCperimeter = request.form['PCperimeter']
                PCarea = request.form['PCarea']
                PCsmoothness = request.form['PCsmoothness']
                PCcompactness = request.form['PCcompactness']

                prd = loadedmodel.predict([[float(PCperimeter), float(PCarea), float(PCsmoothness), float(PCcompactness)]])

                if prd[0] == 1:
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Prostate Cancer"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            PCperimeter = request.form['PCperimeter']
            PCarea = request.form['PCarea']
            PCsmoothness = request.form['PCsmoothness']
            PCcompactness = request.form['PCcompactness']

            prd = loadedmodel.predict([[float(PCperimeter), float(PCarea), float(PCsmoothness), float(PCcompactness)]])

            if prd[0] == 1:
                result = "Positive"
            else:
                result = "Negative"
            disease = "Prostate Cancer"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/MSdiagnosis', methods=['POST'])
def MSdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Multiple Sclerosis", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        pathForModel = open('./static/model/' + modelName, 'rb')
        loadedmodel = pickle.load(pathForModel)

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                age = request.form['LS_Age']
                alt = request.form['LS_ALT']
                ldh = request.form['LS_LDH']
                creatinine = request.form['LS_Creatinine']
                bun = request.form['LS_BUN']
                TotalBilirubin = request.form['LS_TotalBilirubin']
                GGT = request.form['LS_GGT']
                Alk_Phos = request.form['LS_Alk_Phos']
                AST = request.form['LS_AST']
                Platelet = request.form['LS_Platelet']
                BP_Systolic = request.form['LS_BP_Systolic']

                prd = loadedmodel.predict([[float(age), float(alt), float(ldh), float(creatinine), float(bun),
                                            float(TotalBilirubin), float(GGT), float(Alk_Phos), float(AST),
                                            float(Platelet), float(BP_Systolic)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Multiple Sclerosis',
                    'prediction': result,
                    'accuracy': accuracy
                }

                return prediction
            else:
                age = request.form['ageMS']
                alt = request.form['ALTMS']
                ldh = request.form['LDHMS']
                creatinine = request.form['CreatinineMS']
                bun = request.form['BUNMS']
                TotalBilirubin = request.form['TBiliMS']
                GGT = request.form['GGTMS']
                Alk_Phos = request.form['AlkMS']
                AST = request.form['ASTMS']
                Platelet = request.form['PlateletMS']
                BP_Systolic = request.form['BPSysMS']

                prd = loadedmodel.predict([[float(age), float(alt), float(ldh), float(creatinine), float(bun),
                                            float(TotalBilirubin), float(GGT), float(Alk_Phos), float(AST),
                                            float(Platelet),
                                            float(BP_Systolic)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Multiple Sclerosis"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            age = request.form['ageMS']
            alt = request.form['ALTMS']
            ldh = request.form['LDHMS']
            creatinine = request.form['CreatinineMS']
            bun = request.form['BUNMS']
            TotalBilirubin = request.form['TBiliMS']
            GGT = request.form['GGTMS']
            Alk_Phos = request.form['AlkMS']
            AST = request.form['ASTMS']
            Platelet = request.form['PlateletMS']
            BP_Systolic = request.form['BPSysMS']

            prd = loadedmodel.predict([[float(age), float(alt), float(ldh), float(creatinine), float(bun),
                                        float(TotalBilirubin), float(GGT), float(Alk_Phos), float(AST),
                                        float(Platelet),
                                        float(BP_Systolic)]])


            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Multiple Sclerosis"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/ADdiagnosis', methods=['POST'])
def ADdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Alzheimer", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                genderAD = request.form['LS_Sex']
                AgeAD = request.form['LS_Age']
                PulseAD = request.form['LS_Pulse']
                Respiratory_RateAD = request.form['LS_RespiratoryRate']
                BP_DiastolicAD = request.form['LS_BP_Diastolic']
                wbcAD = request.form['LS_WBCcount']
                rbcAD = request.form['LS_RBCcount']
                HemoglobinAD = request.form['LS_Hemoglobin']
                HematocritAD = request.form['LS_Hematocrit']
                MCVAD = request.form['LS_MCV']
                MCHAD = request.form['LS_MCH']
                RDWAD = request.form['LS_RDW']
                MPVAD = request.form['LS_MPV']

                if genderAD == "Male":
                    genderAD = 0
                else:
                    genderAD = 1

                prd = loadedmodel.predict([[float(genderAD), float(AgeAD), float(PulseAD), float(Respiratory_RateAD),
                                            float(BP_DiastolicAD), float(wbcAD), float(rbcAD),
                                            float(HemoglobinAD), float(HematocritAD), float(MCVAD), float(MCHAD),
                                            float(RDWAD), float(MPVAD)]])
                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Alzheimer',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                genderAD = request.form['genderAD']
                AgeAD = request.form['AgeAD']
                PulseAD = request.form['PulseAD']
                Respiratory_RateAD = request.form['Respiratory_RateAD']
                BP_DiastolicAD = request.form['BP_DiastolicAD']
                wbcAD = request.form['wbcAD']
                rbcAD = request.form['rbcAD']
                HemoglobinAD = request.form['HemoglobinAD']
                HematocritAD = request.form['HematocritAD']
                MCVAD = request.form['MCVAD']
                MCHAD = request.form['MCHAD']
                RDWAD = request.form['RDWAD']
                MPVAD = request.form['MPVAD']

                if genderAD == "Male":
                    genderAD = 0
                else:
                    genderAD = 1

                prd = loadedmodel.predict([[float(genderAD), float(AgeAD), float(PulseAD), float(Respiratory_RateAD),
                                            float(BP_DiastolicAD), float(wbcAD), float(rbcAD),
                                            float(HemoglobinAD), float(HematocritAD), float(MCVAD), float(MCHAD),
                                            float(RDWAD), float(MPVAD)]])

                if prd[0] == "1":

                    result = "Positive"

                else:
                    result = "Negative"
                disease = "Alzheimer Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            genderAD = request.form['genderAD']
            AgeAD = request.form['AgeAD']
            PulseAD = request.form['PulseAD']
            Respiratory_RateAD = request.form['Respiratory_RateAD']
            BP_DiastolicAD = request.form['BP_DiastolicAD']
            wbcAD = request.form['wbcAD']
            rbcAD = request.form['rbcAD']
            HemoglobinAD = request.form['HemoglobinAD']
            HematocritAD = request.form['HematocritAD']
            MCVAD = request.form['MCVAD']
            MCHAD = request.form['MCHAD']
            RDWAD = request.form['RDWAD']
            MPVAD = request.form['MPVAD']

            if genderAD == "Male":
                genderAD = 0
            else:
                genderAD = 1

            prd = loadedmodel.predict([[float(genderAD), float(AgeAD), float(PulseAD), float(Respiratory_RateAD),
                                        float(BP_DiastolicAD), float(wbcAD), float(rbcAD),
                                        float(HemoglobinAD), float(HematocritAD), float(MCVAD), float(MCHAD),
                                        float(RDWAD), float(MPVAD)]])

            if prd[0] == "1":

                result = "Positive"

            else:
                result = "Negative"
            disease = "Alzheimer Disease"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

#it is not in the system
@app.route('/ADHDdiagnosis', methods=['POST'])
def ADHDdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "ADHD", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        pathForModel = open('./static/model/' + modelName, 'rb')
        loadedmodel = pickle.load(pathForModel)

        genderADHD = request.form['genderADHD']
        ageADHD = request.form['ageADHD']
        indADHD = request.form['indADHD']
        inatADHD = request.form['inatADHD']
        hypADHD = request.form['hypADHD']
        vqADHD = request.form['vqADHD']
        perfADHD = request.form['perfADHD']
        fqADHD = request.form['fqADHD']

        if genderADHD == "Male":
            genderADHD = 1
        else:
            genderADHD = 2

        prd = loadedmodel.predict([[float(ageADHD), float(indADHD), float(inatADHD), float(hypADHD), float(vqADHD),
                                    float(perfADHD), float(fqADHD)]])
        if prd[0] == "1":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Attention Deficit Hyperactivity Disorder (ADHD) "
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']

            elif session['role'] == 'registered user':
                ID = session['username']
            prediction = {
                'disease': 'ADHD',
                'prediction': result,
                'accuracy': accuracy
            }

            new_result = {
                'user_id': ID,
                'test_date': str(date.today()),
                'predicted_disease': prediction
            }

        db.test.insert_one(new_result)

    finally:
        client.close()
    return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

#it is not in the system
@app.route('/BCdiagnosis', methods=['POST'])
def BCdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Breast Cancer", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        pathForModel = open('./static/model/' + modelName, 'rb')
        loadedmodel = pickle.load(pathForModel)

        AgeBC = request.form['AgeBC']
        BMIBC = request.form['BMIBC']
        GlucoseBC = request.form['GlucoseBC']
        HOMABC = request.form['HOMABC']
        ResistinBC = request.form['ResistinBC']

        prd = loadedmodel.predict([[float(AgeBC), float(BMIBC), float(GlucoseBC), float(HOMABC), float(ResistinBC)]])
        if prd[0] == "1":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Breast Cancer"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']

            elif session['role'] == 'registered user':
                ID = session['username']

            prediction = {
                'disease': 'Breast Cancer',
                'prediction': result,
                'accuracy': accuracy
            }

            new_result = {
                'user_id': ID,
                'test_date': str(date.today()),
                'predicted_disease': prediction
            }

        db.test.insert_one(new_result)

    finally:
        client.close()
    return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

@app.route('/Glaucomadiagnosis', methods=['POST'])
def Glaucomadiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Glaucoma", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'medical specialist':
                atG = request.form['LS_at']
                eanG = request.form['LS_ean']
                mhciG = request.form['LS_mhci']
                vasiG = request.form['LS_vasi']
                vargG = request.form['LS_varg']
                varsG = request.form['LS_vars']
                tmiG = request.form['LS_tmi']

                prd = loadedmodel.predict([[float(atG), float(eanG), float(mhciG), float(vasiG),
                                            float(vargG), float(varsG), float(tmiG)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                prediction = {
                    'disease': 'Glaucoma',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                atG = request.form['atG']
                eanG = request.form['eanG']
                mhciG = request.form['mhciG']
                vasiG = request.form['vasiG']
                vargG = request.form['vargG']
                varsG = request.form['varsG']
                tmiG = request.form['tmiG']

                prd = loadedmodel.predict([[float(atG), float(eanG), float(mhciG), float(vasiG),
                                            float(vargG), float(varsG), float(tmiG)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Glaucoma Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            atG = request.form['atG']
            eanG = request.form['eanG']
            mhciG = request.form['mhciG']
            vasiG = request.form['vasiG']
            vargG = request.form['vargG']
            varsG = request.form['varsG']
            tmiG = request.form['tmiG']

            prd = loadedmodel.predict([[float(atG), float(eanG), float(mhciG), float(vasiG),
                                        float(vargG), float(varsG), float(tmiG)]])

            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Glaucoma Disease"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/LCdiagnosis', methods=['POST'])
def LCdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Lung Cancer", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                genderLC = request.form['LS_Sex']
                ageLC = request.form['LS_Age']
                smokingLC = request.form['LS_smoking']
                yellow_FingersLC = request.form['LS_yellowFingers']
                anxietyLC = request.form['LS_anxiety']
                peer_PressureLC = request.form['LS_peerPressure']
                chronic_DiseaseLC = request.form['LS_chronicDisease']
                fatigueLC = request.form['LS_fatigue']
                allergyLC = request.form['LS_allergy']
                wheezingLC = request.form['LS_wheezing']
                alcoholLC = request.form['LS_alcohol']
                coughingLC = request.form['LS_coughing']
                shortness_of_BreathLC = request.form['LS_shortness_of_Breath']
                swallowing_DifficultyLC = request.form['LS_swallowing_Difficulty']
                chest_PainLC = request.form['LS_chest_Pain']

                if genderLC == "Male":
                    genderLC = 0
                else:
                    genderLC = 1

                prd = loadedmodel.predict([[float(genderLC), float(ageLC), float(smokingLC), float(yellow_FingersLC),
                                            float(anxietyLC), float(wheezingLC),
                                            float(peer_PressureLC), float(chronic_DiseaseLC), float(fatigueLC),
                                            float(allergyLC), float(coughingLC), float(alcoholLC),
                                            float(shortness_of_BreathLC), float(swallowing_DifficultyLC),
                                            float(chest_PainLC)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                prediction = {
                    'disease': 'Lung Cancer',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                genderLC = request.form['genderLC']
                ageLC = request.form['ageLC']
                smokingLC = request.form['smokingLC']
                yellow_FingersLC = request.form['yellow_FingersLC']
                anxietyLC = request.form['anxietyLC']
                peer_PressureLC = request.form['peer_PressureLC']
                chronic_DiseaseLC = request.form['chronic_DiseaseLC']
                fatigueLC = request.form['fatigueLC']
                allergyLC = request.form['allergyLC']
                wheezingLC = request.form['wheezingLC']
                alcoholLC = request.form['alcoholLC']
                coughingLC = request.form['coughingLC']
                shortness_of_BreathLC = request.form['shortness_of_BreathLC']
                swallowing_DifficultyLC = request.form['swallowing_DifficultyLC']
                chest_PainLC = request.form['chest_PainLC']

                if genderLC == "Male":
                    genderLC = 0
                else:
                    genderLC = 1

                prd = loadedmodel.predict([[float(genderLC), float(ageLC), float(smokingLC), float(yellow_FingersLC),
                                            float(anxietyLC), float(wheezingLC),
                                            float(peer_PressureLC), float(chronic_DiseaseLC), float(fatigueLC),
                                            float(allergyLC), float(coughingLC), float(alcoholLC),
                                            float(shortness_of_BreathLC), float(swallowing_DifficultyLC),
                                            float(chest_PainLC)]])

                if prd[0] == "1":

                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Lung Cancer Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            genderLC = request.form['genderLC']
            ageLC = request.form['ageLC']
            smokingLC = request.form['smokingLC']
            yellow_FingersLC = request.form['yellow_FingersLC']
            anxietyLC = request.form['anxietyLC']
            peer_PressureLC = request.form['peer_PressureLC']
            chronic_DiseaseLC = request.form['chronic_DiseaseLC']
            fatigueLC = request.form['fatigueLC']
            allergyLC = request.form['allergyLC']
            wheezingLC = request.form['wheezingLC']
            alcoholLC = request.form['alcoholLC']
            coughingLC = request.form['coughingLC']
            shortness_of_BreathLC = request.form['shortness_of_BreathLC']
            swallowing_DifficultyLC = request.form['swallowing_DifficultyLC']
            chest_PainLC = request.form['chest_PainLC']

            if genderLC == "Male":
                genderLC = 0
            else:
                genderLC = 1

            prd = loadedmodel.predict([[float(genderLC), float(ageLC), float(smokingLC), float(yellow_FingersLC),
                                        float(anxietyLC), float(wheezingLC),
                                        float(peer_PressureLC), float(chronic_DiseaseLC), float(fatigueLC),
                                        float(allergyLC), float(coughingLC), float(alcoholLC),
                                        float(shortness_of_BreathLC), float(swallowing_DifficultyLC),
                                        float(chest_PainLC)]])

            if prd[0] == "1":

                result = "Positive"
            else:
                result = "Negative"
            disease = "Lung Cancer Disease"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
    finally:
        client.close()

@app.route('/CervCdiagnosis', methods=['POST'])
def CervicalCancerdiagnosis():
    try:

        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Cervical Cancer", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                NumOfDiag = request.form['LS_NumOfDiag']
                STD_Condylomatosis = request.form['LS_Condylomatosis']
                StdSyphilis = request.form['LS_StdSyphilis']
                STD_HIV = request.form['LS_HIV']
                STD_HPV = request.form['LS_STD_HPV']
                CC_Dx = request.form['LS_Dx']
                CC_DxCIN = request.form['LS_DxCIN']
                CC_DxHPV = request.form['LS_DxHPV']

                if STD_Condylomatosis == "Yes":
                    STD_Condylomatosis = 1
                else:
                    STD_Condylomatosis = 0

                if StdSyphilis == "Yes":
                    StdSyphilis = 1
                else:
                    StdSyphilis = 0

                if STD_HIV == "Yes":
                    STD_HIV = 1
                else:
                    STD_HIV = 0

                if STD_HPV == "Yes":
                    STD_HPV = 1
                else:
                    STD_HPV = 0

                if CC_Dx == "Yes":
                    CC_Dx = 1
                else:
                    CC_Dx = 0

                if CC_DxCIN == "Yes":
                    CC_DxCIN = 1
                else:
                    CC_DxCIN = 0

                if CC_DxHPV == "Yes":
                    CC_DxHPV = 1
                else:
                    CC_DxHPV = 0

                prd = loadedmodel.predict([[float(STD_Condylomatosis), float(StdSyphilis), float(STD_HIV),
                                            float(STD_HPV), float(CC_DxCIN), float(CC_DxHPV), float(CC_Dx),
                                            float(NumOfDiag)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Cervical Cancer',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                NumOfDiag = request.form['NODCervC']
                STD_Condylomatosis = request.form['CondoCervC']
                StdSyphilis = request.form['SyphCervC']
                STD_HIV = request.form['HIVCervC']
                STD_HPV = request.form['HPVCervC']
                CC_Dx = request.form['DxCervC']
                CC_DxCIN = request.form['DxcinCervC']
                CC_DxHPV = request.form['DxhpvCervC']

                if STD_Condylomatosis == "YES":
                    STD_Condylomatosis = 1
                else:
                    STD_Condylomatosis = 0

                if StdSyphilis == "YES":
                    StdSyphilis = 1
                else:
                    StdSyphilis = 0

                if STD_HIV == "YES":
                    STD_HIV = 1
                else:
                    STD_HIV = 0

                if STD_HPV == "YES":
                    STD_HPV = 1
                else:
                    STD_HPV = 0

                if CC_Dx == "YES":
                    CC_Dx = 1
                else:
                    CC_Dx = 0

                if CC_DxCIN == "YES":
                    CC_DxCIN = 1
                else:
                    CC_DxCIN = 0

                if CC_DxHPV == "YES":
                    CC_DxHPV = 1
                else:
                    CC_DxHPV = 0

                prd = loadedmodel.predict([[float(STD_Condylomatosis), float(StdSyphilis), float(STD_HIV),
                                            float(STD_HPV), float(CC_DxCIN), float(CC_DxHPV), float(CC_Dx),
                                            float(NumOfDiag)]])
                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Cervical Cancer"

                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            NumOfDiag = request.form['NODCervC']
            STD_Condylomatosis = request.form['CondoCervC']
            StdSyphilis = request.form['SyphCervC']
            STD_HIV = request.form['HIVCervC']
            STD_HPV = request.form['HPVCervC']
            CC_Dx = request.form['DxCervC']
            CC_DxCIN = request.form['DxcinCervC']
            CC_DxHPV = request.form['DxhpvCervC']

            if STD_Condylomatosis == "YES":
                STD_Condylomatosis = 1
            else:
                STD_Condylomatosis = 0

            if StdSyphilis == "YES":
                StdSyphilis = 1
            else:
                StdSyphilis = 0

            if STD_HIV == "YES":
                STD_HIV = 1
            else:
                STD_HIV = 0

            if STD_HPV == "YES":
                STD_HPV = 1
            else:
                STD_HPV = 0

            if CC_Dx == "YES":
                CC_Dx = 1
            else:
                CC_Dx = 0

            if CC_DxCIN == "YES":
                CC_DxCIN = 1
            else:
                CC_DxCIN = 0

            if CC_DxHPV == "YES":
                CC_DxHPV = 1
            else:
                CC_DxHPV = 0

            prd = loadedmodel.predict([[float(STD_Condylomatosis), float(StdSyphilis), float(STD_HIV),
                                        float(STD_HPV), float(CC_DxCIN), float(CC_DxHPV), float(CC_Dx),
                                        float(NumOfDiag)]])
            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Cervical Cancer"

            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/HCVdiagnosis', methods=['POST'])
def HepCdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Hepatitis C", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                age = request.form['LS_Age']
                TotP = request.form['LS_TotalProtein']
                TotB = request.form['LS_TotalBilirubin']
                DirB = request.form['LS_DirectBilirubin']
                GGT = request.form['LS_GGT']
                AlkPh = request.form['LS_Alk_Phos']
                Lymph = request.form['LS_Lymphocyte']
                NueAbs = request.form['LS_NeuGran_Abs']
                Plat = request.form['LS_Platelet']
                Baso = request.form['LS_Basophil_Instrument']
                Sys = request.form['LS_BP_Systolic']
                FallRisk = request.form['LS_FallRiskMorse']
                BMI = request.form['LS_BodyMass']
                INR = request.form['LS_IntNormRati']

                #Normalizing the input using (MinMax Normalization)
                age = (float(age) - 13) / (89 - 13)
                TotP = (float(TotP) - 3.1) / (8.9 - 3.1)
                TotB = (float(TotB) - 0.2) / (16.7 - 0.2)
                DirB = (float(DirB) - 0.05) / (11.1 - 0.05)
                GGT = (float(GGT) - 6) / (1296 - 6)
                AlkPh = (float(AlkPh) - 21) / (1881 - 21)
                Lymph = (float(Lymph) - 2) / (58.4 - 2)
                NueAbs = (float(NueAbs) - 0.7) / (12.2 - 0.7)
                Plat = (float(Plat) - 27) / (528 - 27)
                Baso = (float(Baso) - 0) / (2.6 - 0)
                Sys = (float(Sys) - 21.33) / (190 - 21.33)
                FallRisk = (float(FallRisk) - 15.68) / (70 - 15.68)
                BMI = (float(BMI) - 7.8) / (59.8 - 7.8)
                INR = (float(INR) - 0.74) / (5.79)

                prd = loadedmodel.predict([[float(age), float(TotP), float(TotB), float(DirB), float(GGT), float(AlkPh),
                                            float(Lymph), float(NueAbs), float(Plat), float(Baso),
                                            float(Sys), float(FallRisk), float(BMI), float(INR)]])
                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Hepatitis C',
                    'prediction': result,
                    'accuracy': accuracy
                }

                return prediction
            else:
                age = request.form['ageHCV']
                TotP = request.form['TotalProteinHCV']
                TotB = request.form['TotalBilirubinHCV']
                DirB = request.form['DirectBilirubinHCV']
                GGT = request.form['GammaGlutamyltransferaseHCV']
                AlkPh= request.form['AlkalinePhosphataseHCV']
                Lymph= request.form['Lymphocyte_Instrument_HCV']
                NueAbs= request.form['NeutrophilGranulocyte_InstrumentAbsoHCV']
                Plat= request.form['PlateletHCV']
                Baso= request.form['Basophil_Instrument_HCV']
                Sys= request.form['BP_SystolicHCV']
                FallRisk= request.form['FallRisk_MorseHCV']
                BMI= request.form['BodyMassHCV']
                INR= request.form['InternationalNormalizedRatioHCV']

                age = (float(age) - 13) / (89 - 13)
                TotP = (float(TotP) - 3.1) / (8.9 - 3.1)
                TotB = (float(TotB) - 0.2) / (16.7 - 0.2)
                DirB = (float(DirB) - 0.05) / (11.1 - 0.05)
                GGT = (float(GGT) - 6) / (1296 - 6)
                AlkPh = (float(AlkPh) - 21) / (1881 - 21)
                Lymph = (float(Lymph) - 2) / (58.4 - 2)
                NueAbs = (float(NueAbs) - 0.7) / (12.2 - 0.7)
                Plat = (float(Plat) - 27) / (528 - 27)
                Baso = (float(Baso) - 0) / (2.6 - 0)
                Sys = (float(Sys) - 21.33) / (190 - 21.33)
                FallRisk = (float(FallRisk) - 15.68) / (70 - 15.68)
                BMI = (float(BMI) - 7.8) / (59.8 - 7.8)
                INR = (float(INR) - 0.74) / (5.79)

                prd = loadedmodel.predict([[float(age), float(TotP), float(TotB),float(DirB), float(GGT),float(AlkPh),
                                            float(Lymph), float(NueAbs), float(Plat),float(Baso),
                                            float(Sys), float(FallRisk), float(BMI),float(INR)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Hepatitis C"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            age = request.form['ageHCV']
            TotP = request.form['TotalProteinHCV']
            TotB = request.form['TotalBilirubinHCV']
            DirB = request.form['DirectBilirubinHCV']
            GGT = request.form['GammaGlutamyltransferaseHCV']
            AlkPh = request.form['AlkalinePhosphataseHCV']
            Lymph = request.form['Lymphocyte_Instrument_HCV']
            NueAbs = request.form['NeutrophilGranulocyte_InstrumentAbsoHCV']
            Plat = request.form['PlateletHCV']
            Baso = request.form['Basophil_Instrument_HCV']
            Sys = request.form['BP_SystolicHCV']
            FallRisk = request.form['FallRisk_MorseHCV']
            BMI = request.form['BodyMassHCV']
            INR = request.form['InternationalNormalizedRatioHCV']

            age = (float(age) - 13) / (89 - 13)
            TotP = (float(TotP) - 3.1) / (8.9 - 3.1)
            TotB = (float(TotB) - 0.2) / (16.7 - 0.2)
            DirB = (float(DirB) - 0.05) / (11.1 - 0.05)
            GGT = (float(GGT) - 6) / (1296 - 6)
            AlkPh = (float(AlkPh) - 21) / (1881 - 21)
            Lymph = (float(Lymph) - 2) / (58.4 - 2)
            NueAbs = (float(NueAbs) - 0.7) / (12.2 - 0.7)
            Plat = (float(Plat) - 27) / (528 - 27)
            Baso = (float(Baso) - 0) / (2.6 - 0)
            Sys = (float(Sys) - 21.33) / (190 - 21.33)
            FallRisk = (float(FallRisk) - 15.68) / (70 - 15.68)
            BMI = (float(BMI) - 7.8) / (59.8 - 7.8)
            INR = (float(INR) - 0.74) / (5.79)

            prd = loadedmodel.predict([[float(age), float(TotP), float(TotB), float(DirB), float(GGT), float(AlkPh),
                                        float(Lymph), float(NueAbs), float(Plat), float(Baso),
                                        float(Sys), float(FallRisk), float(BMI), float(INR)]])

            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Hepatitis C"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()
        pass

@app.route('/Depdiagnosis', methods=['POST'])
def Depressiondiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Depression", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                age = request.form['LS_Age']
                hhsize = request.form['LS_hhsize']
                edu = request.form['LS_EducationLevel']
                asset_livestock = request.form['LS_ValOfLivestock']
                asset_durable = request.form['LS_ValOfDurabGood']
                asset_savings = request.form['LS_ValSaving']
                asset_land_owned_total = request.form['LS_LandOwned']
                cons_alcohol = request.form['LS_AlcoholConsumed']
                cons_tobacco = request.form['LS_TobaccoConsumed']
                cons_ed = request.form['LS_EduExpenditure']
                ent_nonag_flowcost = request.form['LS_Ent_nonag_flowcost']
                ent_animalstockrev = request.form['LS_Ent_animalstockrev']
                ent_total_cost = request.form['LS_Ent_total_cost']
                fs_adwholed_often = request.form['LS_Fs_adwholed_often']
                nondurable_investment = request.form['LS_Nondurable_investment']
                amount_received_mpesa = request.form['LS_Amount_received_mpesa']
                Married = request.form['LS_Married']
                Children = request.form['LS_Children']
                hh_children = request.form['LS_hh_children']
                ent_nonagbusiness = request.form['LS_Ent_nonagbusiness']
                saved_mpesa = request.form['LS_Saved_mpesa']
                early_survey = request.form['LS_Early_survey']


                if Married == 'Yes':
                    Married = 1
                else:
                    Married = 0

                if ent_nonagbusiness == 'Yes':
                    ent_nonagbusiness = 1
                else:
                    ent_nonagbusiness = 0

                if saved_mpesa == 'Yes':
                    saved_mpesa = 1
                else:
                    saved_mpesa = 0

                if early_survey == 'Yes':
                    early_survey = 1
                else:
                    early_survey = 0

                prd = loadedmodel.predict([[float(age), float(hhsize), float(edu), float(asset_livestock),
                                            float(asset_durable), float(asset_savings), float(asset_land_owned_total),
                                            float(cons_alcohol), float(cons_tobacco), float(cons_ed),
                                            float(ent_nonag_flowcost), float(ent_animalstockrev), float(ent_total_cost),
                                            float(fs_adwholed_often), float(nondurable_investment),
                                            float(amount_received_mpesa), float(Married), float(Children),
                                            float(hh_children), float(ent_nonagbusiness), float(saved_mpesa),
                                            float(early_survey)]])
                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Depression',
                    'prediction': result,
                    'accuracy': accuracy
                }

                return prediction
            else:
                age = request.form['ageDep']
                hhsize = request.form['hhsizeDep']
                edu = request.form['eduDep']
                asset_livestock = request.form['asset_livestockDep']
                asset_durable = request.form['asset_durableDep']
                asset_savings = request.form['asset_savingsDep']
                asset_land_owned_total = request.form['asset_land_owned_totalDep']
                cons_alcohol = request.form['cons_alcoholDep']
                cons_tobacco = request.form['cons_tobaccoDep']
                cons_ed = request.form['cons_edDep']
                ent_nonag_flowcost = request.form['ent_nonag_flowcostDep']
                ent_animalstockrev = request.form['ent_animalstockrevDep']
                ent_total_cost = request.form['ent_total_costDep']
                fs_adwholed_often = request.form['fs_adwholed_oftenDep']
                nondurable_investment = request.form['nondurable_investmentDep']
                amount_received_mpesa = request.form['amount_received_mpesaDep']
                Married = request.form['MarriedDep']
                Children = request.form['ChildrenDep']
                hh_children = request.form['hh_childrenDep']
                ent_nonagbusiness = request.form['ent_nonagbusinessDep']
                saved_mpesa = request.form['saved_mpesaDep']
                early_survey = request.form['early_surveyDep']


                if Married == 'Yes':
                    Married=1
                else:
                    Married=0

                if ent_nonagbusiness == 'Yes':
                    ent_nonagbusiness=1
                else:
                    ent_nonagbusiness=0

                if saved_mpesa == 'Yes':
                    saved_mpesa=1
                else:
                    saved_mpesa=0

                if early_survey == 'Yes':
                    early_survey=1
                else:
                    early_survey=0


                prd = loadedmodel.predict([[float(age), float(hhsize), float(edu),float(asset_livestock),
                                            float(asset_durable),float(asset_savings), float(asset_land_owned_total),
                                            float(cons_alcohol), float(cons_tobacco), float(cons_ed),
                                            float(ent_nonag_flowcost),float(ent_animalstockrev), float(ent_total_cost),
                                            float(fs_adwholed_often), float(nondurable_investment),
                                            float(amount_received_mpesa),float(Married), float(Children),
                                            float(hh_children), float(ent_nonagbusiness),float(saved_mpesa),
                                            float(early_survey)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Depression"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            age = request.form['ageDep']
            hhsize = request.form['hhsizeDep']
            edu = request.form['eduDep']
            asset_livestock = request.form['asset_livestockDep']
            asset_durable = request.form['asset_durableDep']
            asset_savings = request.form['asset_savingsDep']
            asset_land_owned_total = request.form['asset_land_owned_totalDep']
            cons_alcohol = request.form['cons_alcoholDep']
            cons_tobacco = request.form['cons_tobaccoDep']
            cons_ed = request.form['cons_edDep']
            ent_nonag_flowcost = request.form['ent_nonag_flowcostDep']
            ent_animalstockrev = request.form['ent_animalstockrevDep']
            ent_total_cost = request.form['ent_total_costDep']
            fs_adwholed_often = request.form['fs_adwholed_oftenDep']
            nondurable_investment = request.form['nondurable_investmentDep']
            amount_received_mpesa = request.form['amount_received_mpesaDep']
            Married = request.form['MarriedDep']
            Children = request.form['ChildrenDep']
            hh_children = request.form['hh_childrenDep']
            ent_nonagbusiness = request.form['ent_nonagbusinessDep']
            saved_mpesa = request.form['saved_mpesaDep']
            early_survey = request.form['early_surveyDep']

            if Married == 'Yes':
                Married = 1
            else:
                Married = 0

            if ent_nonagbusiness == 'Yes':
                ent_nonagbusiness = 1
            else:
                ent_nonagbusiness = 0

            if saved_mpesa == 'Yes':
                saved_mpesa = 1
            else:
                saved_mpesa = 0

            if early_survey == 'Yes':
                early_survey = 1
            else:
                early_survey = 0

            prd = loadedmodel.predict([[float(age), float(hhsize), float(edu), float(asset_livestock),
                                        float(asset_durable), float(asset_savings), float(asset_land_owned_total),
                                        float(cons_alcohol), float(cons_tobacco), float(cons_ed),
                                        float(ent_nonag_flowcost), float(ent_animalstockrev), float(ent_total_cost),
                                        float(fs_adwholed_often), float(nondurable_investment),
                                        float(amount_received_mpesa), float(Married), float(Children),
                                        float(hh_children), float(ent_nonagbusiness), float(saved_mpesa),
                                        float(early_survey)]])

            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Depression"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/LCHdiagnosis', methods=['POST'])
def LCHdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Liver Cirrhosis", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        #modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                gender = request.form['LS_Sex']
                age = request.form['LS_Age']
                ndays = request.form['LS_NDays']
                hepatomegaly = request.form['LS_Hepatomegaly']
                spiders = request.form['LS_Spiders']
                edema = request.form['LS_Edema']
                cholesterol = request.form['LS_Cholesterol']
                copper = request.form['LS_Copper']
                sgot = request.form['LS_SGOT']
                platelet = request.form['LS_Platelet']
                prothrombin = request.form['LS_Prothrombin']
                ascites = request.form['LS_Ascites']
                bilirubin = request.form['LS_SerumBilirubin']
                albumin = request.form['LS_Albumin']
                alk_phos = request.form['LS_Alk_Phos']
                triglycerides = request.form['LS_Triglycerides']
                drug = request.form['LS_Drug']
                status_lc = request.form['LS_StatusLC']

                if hepatomegaly == "Yes":
                    hepatomegaly = 1
                else:
                    hepatomegaly = 0

                if spiders == "Yes":
                    spiders = 1
                else:
                    spiders = 0

                if edema == "Yes":
                    edema = 1
                else:
                    edema = 0

                if ascites == "Yes":
                    ascites = 1
                else:
                    ascites = 0

                if drug == "D-penicillamine":
                    drug = 1
                else:
                    drug = 0

                if status_lc == "death":
                    status_lc = 1
                elif status_lc == "censored":
                    status_lc = 0
                else:
                    status_lc = 2

                if gender == "Male":
                    gender = 0
                else:
                    gender = 1

                prd = loadedmodel.predict([[float(gender), float(age), float(ndays), float(hepatomegaly),
                                            float(spiders), float(edema), float(cholesterol), float(copper), float(sgot),
                                            float(platelet), float(prothrombin), float(ascites), float(bilirubin), float(albumin),
                                            float(alk_phos), float(triglycerides), float(drug), float(status_lc)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Liver Cirrhosis',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                gender = request.form['sexLCH']
                age = request.form['ageLCH']
                ndays = request.form['N_DaysLCH']
                hepatomegaly = request.form['HepatomegalyLCH']
                spiders = request.form['SpidersLCH']
                edema = request.form['EdemaLCH']
                cholesterol = request.form['CholesterolLCH']
                copper = request.form['CopperLCH']
                sgot = request.form['SGOTLCH']
                platelet = request.form['PlateletsLCH']
                prothrombin = request.form['ProthrombinLCH']
                ascites = request.form['AscitesLCH']
                bilirubin = request.form['BilirubinLCH']
                albumin = request.form['AlbuminLCH']
                alk_phos = request.form['Alk_PhosLCH']
                triglycerides = request.form['TriglyceridessLCH']
                drug = request.form['DrugLCH']
                status_lc = request.form['StatusLCH']

                if hepatomegaly == "Yes":
                    hepatomegaly = 1
                else:
                    hepatomegaly = 0

                if spiders == "Yes":
                    spiders = 1
                else:
                    spiders = 0

                if edema == "Yes":
                    edema = 1
                else:
                    edema = 0

                if ascites == "Yes":
                    ascites = 1
                else:
                    ascites = 0

                if drug == "D-penicillamine":
                    drug = 1
                else:
                    drug = 0

                if status_lc == "death":
                    status_lc = 1
                elif status_lc == "censored":
                    status_lc = 0
                else:
                    status_lc = 2

                if gender == "Male":
                    gender = 0
                else:
                    gender = 1

                prd = loadedmodel.predict([[float(gender), float(age), float(ndays), float(hepatomegaly),
                                            float(spiders), float(edema), float(cholesterol), float(copper), float(sgot),
                                            float(platelet), float(prothrombin), float(ascites), float(bilirubin), float(albumin),
                                            float(alk_phos), float(triglycerides), float(drug), float(status_lc)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Liver Cirrhosis"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            gender = request.form['sexLCH']
            age = request.form['ageLCH']
            ndays = request.form['N_DaysLCH']
            hepatomegaly = request.form['HepatomegalyLCH']
            spiders = request.form['SpidersLCH']
            edema = request.form['EdemaLCH']
            cholesterol = request.form['CholesterolLCH']
            copper = request.form['CopperLCH']
            sgot = request.form['SGOTLCH']
            platelet = request.form['PlateletsLCH']
            prothrombin = request.form['ProthrombinLCH']
            ascites = request.form['AscitesLCH']
            bilirubin = request.form['BilirubinLCH']
            albumin = request.form['AlbuminLCH']
            alk_phos = request.form['Alk_PhosLCH']
            triglycerides = request.form['TriglyceridessLCH']
            drug = request.form['DrugLCH']
            status_lc = request.form['StatusLCH']

            if hepatomegaly == "Yes":
                hepatomegaly = 1
            else:
                hepatomegaly = 0

            if spiders == "Yes":
                spiders = 1
            else:
                spiders = 0

            if edema == "Yes":
                edema = 1
            else:
                edema = 0

            if ascites == "Yes":
                ascites = 1
            else:
                ascites = 0

            if drug == "D-penicillamine":
                drug = 1
            else:
                drug = 0

            if status_lc == "death":
                status_lc = 1
            elif status_lc == "censored":
                status_lc = 0
            else:
                status_lc = 2

            if gender == "Male":
                gender = 0
            else:
                gender = 1

            prd = loadedmodel.predict([[float(gender), float(age), float(ndays), float(hepatomegaly),
                                        float(spiders), float(edema), float(cholesterol), float(copper), float(sgot),
                                        float(platelet), float(prothrombin), float(ascites), float(bilirubin),
                                        float(albumin),
                                        float(alk_phos), float(triglycerides), float(drug), float(status_lc)]])

            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Liver Cirrhosis"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/PRDdiagnosis', methods=['POST'])
def PRDdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Parkinsons Disease", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                gender = request.form['LS_Sex']
                age = request.form['LS_Age']
                anion_gap = request.form['LS_Anion_Gap']
                alt = request.form['LS_ALT']
                ldh = request.form['LS_LDH']
                wbc = request.form['LS_WBC']
                rbc = request.form['LS_RBC']
                hemoglobin = request.form['LS_Hemoglobin']
                hematocrit = request.form['LS_Hematocrit']
                sodium = request.form['LS_Sodium']
                potassium = request.form['LS_Potassium']
                chloride = request.form['LS_Chloride']
                carbon_dioxide = request.form['LS_Carbon_Dioxide']
                creatinine = request.form['LS_Creatinine']
                total_protein = request.form['LS_TotalProtein']
                albumin = request.form['LS_Albumin']
                bun = request.form['LS_BUN']
                total_bilirubin = request.form['LS_TotalBilirubin']
                direct_bilirubin = request.form['LS_DirectBilirubin']
                ggt = request.form['LS_GGT']
                mcv = request.form['LS_MCV']
                mch = request.form['LS_MCH']
                mchc = request.form['LS_MCHC']
                alk_phos = request.form['LS_Alk_Phos']
                rdw = request.form['LS_RDW']
                ast = request.form['LS_AST']

                if gender == "Male":
                    gender = 0
                else:
                    gender = 1

                prd = loadedmodel.predict([[float(gender), float(age), float(anion_gap), float(alt),
                                            float(ldh), float(wbc), float(rbc), float(hemoglobin), float(hematocrit),
                                            float(sodium), float(potassium), float(chloride), float(carbon_dioxide), float(creatinine),
                                            float(total_protein), float(albumin), float(bun), float(total_bilirubin), float(direct_bilirubin),
                                            float(ggt), float(mcv), float(mch), float(mchc), float(alk_phos),
                                            float(rdw), float(ast)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Parkinson’s Disease',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                gender = request.form['genderPRD']
                age = request.form['agePRD']
                anion_gap = request.form['Anion_GapPRD']
                alt = request.form['ALTPRD']
                ldh = request.form['LDHPRD']
                wbc = request.form['WBCPRD']
                rbc = request.form['RBCPRD']
                hemoglobin = request.form['HemoglobinPRD']
                hematocrit = request.form['HematocritPRD']
                sodium = request.form['SodiumPRD']
                potassium = request.form['PotassiumPRD']
                chloride = request.form['ChloridePRD']
                carbon_dioxide = request.form['Carbon_DioxidePRD']
                creatinine = request.form['CreatininePRD']
                total_protein = request.form['Total_ProteinPRD']
                albumin = request.form['AlbuminPRD']
                bun = request.form['BUNPRD']
                total_bilirubin = request.form['Total_BilirubinPRD']
                direct_bilirubin = request.form['Direct_BilirubinPRD']
                ggt = request.form['GGTPRD']
                mcv = request.form['MCVPRD']
                mch = request.form['MCHPRD']
                mchc = request.form['MCHCPRD']
                alk_phos = request.form['ALKPPRD']
                rdw = request.form['RDWPRD']
                ast = request.form['ASTPRD']

                if gender == "Male":
                    gender = 0
                else:
                    gender = 1

                prd = loadedmodel.predict([[float(gender), float(age), float(anion_gap), float(alt),
                                            float(ldh), float(wbc), float(rbc), float(hemoglobin), float(hematocrit),
                                            float(sodium), float(potassium), float(chloride), float(carbon_dioxide), float(creatinine),
                                            float(total_protein), float(albumin), float(bun), float(total_bilirubin), float(direct_bilirubin),
                                            float(ggt), float(mcv), float(mch), float(mchc), float(alk_phos),
                                            float(rdw), float(ast)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Parkinson’s Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            gender = request.form['genderPRD']
            age = request.form['agePRD']
            anion_gap = request.form['Anion_GapPRD']
            alt = request.form['ALTPRD']
            ldh = request.form['LDHPRD']
            wbc = request.form['WBCPRD']
            rbc = request.form['RBCPRD']
            hemoglobin = request.form['HemoglobinPRD']
            hematocrit = request.form['HematocritPRD']
            sodium = request.form['SodiumPRD']
            potassium = request.form['PotassiumPRD']
            chloride = request.form['ChloridePRD']
            carbon_dioxide = request.form['Carbon_DioxidePRD']
            creatinine = request.form['CreatininePRD']
            total_protein = request.form['Total_ProteinPRD']
            albumin = request.form['AlbuminPRD']
            bun = request.form['BUNPRD']
            total_bilirubin = request.form['Total_BilirubinPRD']
            direct_bilirubin = request.form['Direct_BilirubinPRD']
            ggt = request.form['GGTPRD']
            mcv = request.form['MCVPRD']
            mch = request.form['MCHPRD']
            mchc = request.form['MCHCPRD']
            alk_phos = request.form['ALKPPRD']
            rdw = request.form['RDWPRD']
            ast = request.form['ASTPRD']

            if gender == "Male":
                gender = 0
            else:
                gender = 1

            prd = loadedmodel.predict([[float(gender), float(age), float(anion_gap), float(alt),
                                        float(ldh), float(wbc), float(rbc), float(hemoglobin), float(hematocrit),
                                        float(sodium), float(potassium), float(chloride), float(carbon_dioxide),
                                        float(creatinine),
                                        float(total_protein), float(albumin), float(bun), float(total_bilirubin),
                                        float(direct_bilirubin),
                                        float(ggt), float(mcv), float(mch), float(mchc), float(alk_phos),
                                        float(rdw), float(ast)]])

            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Parkinson’s Disease"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/COPDdiagnosis', methods=['POST'])
def COPDdiagnosis():
    try:

        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Chronic Obstructive Pulmonary Disease", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                sex = request.form['LS_Sex']
                age = request.form['LS_Age']
                smoking = request.form['LS_smoking']
                imagery_part_min = request.form['LS_Imagery_part_min']
                imagery_part_avg = request.form['LS_Imagery_part_avg']
                real_part_min = request.form['LS_Real_part_min']
                real_part_avg = request.form['LS_Real_part_avg']

        #gender
                if sex == "Male":
                    sex = 0
                else:
                    sex = 1

                prd = loadedmodel.predict([[float(sex), float(age), float(smoking), float(imagery_part_min),
                                            float(imagery_part_avg), float(real_part_min), float(real_part_avg)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Chronic Obstructive Pulmonary Disease',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                sex = request.form['genderCOPD']
                age = request.form['ageCOPD']
                smoking = request.form['SmokingCOPD']
                imagery_part_min = request.form['Imagery_part_minCOPD']
                imagery_part_avg = request.form['Imagery_part_avgCOPD']
                real_part_min = request.form['Real_part_minCOPD']
                real_part_avg = request.form['Real_part_avgCOPD']

                if sex == "Male":
                    sex = 0
                else:
                    sex = 1

                prd = loadedmodel.predict([[float(sex), float(age), float(smoking), float(imagery_part_min),
                                            float(imagery_part_avg), float(real_part_min), float(real_part_avg)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Chronic Obstructive Pulmonary Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            sex = request.form['genderCOPD']
            age = request.form['ageCOPD']
            smoking = request.form['SmokingCOPD']
            imagery_part_min = request.form['Imagery_part_minCOPD']
            imagery_part_avg = request.form['Imagery_part_avgCOPD']
            real_part_min = request.form['Real_part_minCOPD']
            real_part_avg = request.form['Real_part_avgCOPD']

            if sex == "Male":
                sex = 0
            else:
                sex = 1

            prd = loadedmodel.predict([[float(sex), float(age), float(smoking), float(imagery_part_min),
                                        float(imagery_part_avg), float(real_part_min), float(real_part_avg)]])

            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Chronic Obstructive Pulmonary Disease"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()
#diagnose hypertension
@app.route('/Hyperdiagnosis', methods=['POST'])
def Hyperdiagnosis():
    try:
        db, client = connection()
        print("Database connected successfully")

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Hypertension Disease", "Active": 1}
        result = db.model.find_one(query)
        if result is None:
            print("No active model found for Hypertension Disease")
            return "Error: No model found", 400
        print("Model found: ", result)

       # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path
        print("Model loaded successfully")
        print("Route hit. Request data:", request.data)
        print("Form data:", request.form)
        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                try:
                    age = float(request.form['LS_Age'])
                except ValueError:
                    return jsonify({'error': 'Invalid input for age'}), 400
                hemoglobin = request.form['LS_Hemoglobin']
                protien = request.form['LS_ProtienUA']
                pH_urine= request.form['LS_pH_Urine']
                calcium = request.form['LS_Calcium']
                glucose = request.form['LS_Glucose']
                hdl= request.form['LS_HDL']
                ldlhdl= request.form['LS_HDL_LDL']
                ldl_chol= request.form['LS_LDL_Cholesterol']
                cholesterol= request.form['LS_Cholesterol']


                prd = loadedmodel.predict([[float(protien),float(age),float(pH_urine),float(calcium),float(hemoglobin),float(hdl)
                                             ,float(ldlhdl), float(cholesterol), float(glucose), float(ldl_chol)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Hypertension Disease',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                try:
                    age = float(request.form.get('ageHyper'))
                except ValueError:
                    return jsonify({'error': 'Invalid input for age'}), 400
                hemoglobin = request.form.get('HemoHyper')
                protien = request.form.get('ProteinHyper')
                pH_urine= request.form.get('phHyper')
                calcium = request.form[('CalciumHyper')]
                glucose = request.form[('GlucoseHyper')]
                hdl= request.form.get('HDLHyper')
                ldlhdl= request.form.get('HDLLDLHyper')
                ldl_chol= request.form.get('LDLColHyper')
                cholesterol= request.form.get('CholesterolHyper')
                print(protien,age,pH_urine,calcium,hemoglobin,hdl,ldlhdl,cholesterol,glucose,ldl_chol)
                prd = loadedmodel.predict([[float(protien),float(age),float(pH_urine),float(calcium),float(hemoglobin),float(hdl)
                                             ,float(ldlhdl), float(cholesterol), float(glucose), float(ldl_chol)]])
                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Hypertension Disease"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            try:
                age = float(request.form.get('ageHyper'))
            except ValueError:
                return jsonify({'error': 'Invalid input for age'}), 400
            hemoglobin = request.form.get('HemoHyper')
            protien = request.form.get('ProteinHyper')
            pH_urine= request.form.get('phHyper')
            calcium = request.form['CalciumHyper']
            glucose = request.form['GlucoseHyper']
            hdl= request.form.get('HDLHyper')
            ldlhdl= request.form.get('HDLLDLHyper')
            ldl_chol= request.form.get('LDLColHyper')
            cholesterol= request.form.get('CholesterolHyper')
            print(protien,age,pH_urine,calcium,hemoglobin,hdl,ldlhdl,cholesterol,glucose,ldl_chol)

            prd = loadedmodel.predict([[float(protien),float(age),float(pH_urine),float(calcium),float(hemoglobin),float(hdl)
                                            ,float(ldlhdl), float(cholesterol), float(glucose), float(ldl_chol)]])
            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Hypertension Disease"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
    except Exception as e:
        print("An error occurred: ", str(e))
        return str(e), 500
    finally:
        client.close()
@app.route('/Skindiagnosis', methods=['POST'])
def SkinDiagnosis():
    try:
        db, client = connection()
        print("Database connected successfully")

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Skin Cancer", "Active": 1}
        result = db.model.find_one(query)
        if result is None:
            print("No active model found for Skin Cancer")
            return "Error: No model found", 400
        print("Model found: ", result)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)
        
        # loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path
        loadedmodel = joblib.load('./static/model/' + modelName)
        print("Model loaded successfully")
        print("Route hit. Request data:", request.data)
        print("Form data:", request.form)
        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                try:
                    age = float(request.form['LS_Age'])
                except ValueError:
                    return jsonify({'error': 'Invalid input for age'}), 400
                smoke= request.form['LS_smoking']
                pesticide = request.form['LS_pesticide']
                sex = request.form['LS_Sex']
                skin_cancer_history = request.form['LS_skin_cancer_hist']
                cancer_history	= request.form['LS_cancer_hist']
                has_piped_water = request.form['LS_has_piped_water']
                has_sewage_system = request.form['LS_has_sewage_system']
                
                #gender
                if sex == "Male":
                    sex = 0
                else:
                    sex = 1

                if smoke == "Yes":
                    smoke = 1
                else:
                    smoke = 0
                
                if pesticide == "Yes":
                    pesticide = 1
                else:
                    pesticide = 0
                    
                if skin_cancer_history == "Yes":
                    skin_cancer_history = 1
                else:
                    skin_cancer_history = 0
                
                if cancer_history == "Yes":
                    cancer_history = 1
                else:
                    cancer_history = 0
                    
                if has_piped_water == "Yes":
                    has_piped_water = 1
                else:
                    has_piped_water = 0 
                    
                if has_sewage_system == "Yes":
                    has_sewage_system = 1
                else:
                    has_sewage_system = 0
                    
                print(smoke,age,pesticide,sex,skin_cancer_history,cancer_history,has_piped_water,has_sewage_system)
    
                prd = loadedmodel.predict([[bool(smoke),float(age),bool(pesticide),float(sex),bool(skin_cancer_history),bool(cancer_history)
                                            ,bool(has_piped_water), bool(has_sewage_system)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Skin Cancer',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                try:
                    age = float(request.form['ageSkin'])
                except ValueError:
                    return jsonify({'error': 'Invalid input for age'}), 400
                smoke= request.form['SkinSmoking']
                pesticide = request.form['pesticide']
                sex = request.form['SexSkin']
                skin_cancer_history = request.form['skin_cancer_hist']
                cancer_history	= request.form['cancer_hist']
                has_piped_water = request.form['has_piped_water']
                has_sewage_system = request.form['has_sewage_system']
                
                #gender
                if sex == "Male":
                    sex = 0
                else:
                    sex = 1

                if smoke == "Yes":
                    smoke = 1
                else:
                    smoke = 0
                
                if pesticide == "Yes":
                    pesticide = 1
                else:
                    pesticide = 0
                    
                if skin_cancer_history == "Yes":
                    skin_cancer_history = 1
                else:
                    skin_cancer_history = 0
                
                if cancer_history == "Yes":
                    cancer_history = 1
                else:
                    cancer_history = 0
                    
                if has_piped_water == "Yes":
                    has_piped_water = 1
                else:
                    has_piped_water = 0 
                    
                if has_sewage_system == "Yes":
                    has_sewage_system = 1
                else:
                    has_sewage_system = 0
                    
                print(smoke,age,pesticide,sex,skin_cancer_history,cancer_history,has_piped_water,has_sewage_system)
                prd = loadedmodel.predict([[bool(smoke),float(age),bool(pesticide),float(sex),
                                            bool(skin_cancer_history),bool(cancer_history),bool(has_piped_water), bool(has_sewage_system)]])
                
                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Skin Cancer"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            try:
                age = float(request.form['ageSkin'])
            except ValueError:
                return jsonify({'error': 'Invalid input for age'}), 400
            smoke= request.form['SkinSmoking']
            pesticide = request.form['pesticide']
            sex = request.form['SexSkin']
            skin_cancer_history = request.form['skin_cancer_hist']
            cancer_history	= request.form['cancer_hist']
            has_piped_water = request.form['has_piped_water']
            has_sewage_system = request.form['has_sewage_system']
            
            #gender
            if sex == "Male":
                sex = 0
            else:
                sex = 1

            if smoke == "Yes":
                smoke = 1
            else:
                smoke = 0
            
            if pesticide == "Yes":
                pesticide = 1
            else:
                pesticide = 0
                
            if skin_cancer_history == "Yes":
                skin_cancer_history = 1
            else:
                skin_cancer_history = 0
            
            if cancer_history == "Yes":
                cancer_history = 1
            else:
                cancer_history = 0
                
            if has_piped_water == "Yes":
                has_piped_water = 1
            else:
                has_piped_water = 0 
                
            if has_sewage_system == "Yes":
                has_sewage_system = 1
            else:
                has_sewage_system = 0
                
            print(smoke,age,pesticide,sex,skin_cancer_history,cancer_history,has_piped_water,has_sewage_system)
            prd = loadedmodel.predict([[bool(smoke),float(age),bool(pesticide),float(sex),
                                        bool(skin_cancer_history),bool(cancer_history),bool(has_piped_water), bool(has_sewage_system)]])
        
            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Skin Cancer"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
    except Exception as e:
        print("An error occurred: ", str(e))
        return str(e), 500
    finally:
        client.close()



@app.route('/Colorectaldiagnosis', methods=['POST'])
def ColorectDiagnosis():
    try:
        db, client = connection()
        print("Database connected successfully")

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Colorectal Cancer", "Active": 1}
        result = db.model.find_one(query)
        if result is None:
            print("No active model found for Colorectal Cancer")
            return "Error: No model found", 400
        print("Model found: ", result)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        # loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # check Path
        loadedmodel = joblib .load('./static/model/' + modelName) # check Path
        print("Model loaded successfully")
        print("Route hit. Request data:", request.data)
        print("Form data:", request.form)
        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                try:
                    age = float(request.form['LS_Age'])
                except ValueError:
                    return jsonify({'error': 'Invalid input for age'}), 400
                p16540= request.form['LS_p16540']
                p16580 = request.form['LS_p16580']
                mdm2 = request.form['LS_mdm2']
                GAL3 = request.form['LS_GAL3']
                TIM1 = request.form['LS_TIM1']
                
                p16540cc =0
                p16540gc = 0
                p16540gg = 0
                
                p16580cc = 0
                p16580ct = 0
                p16580tt = 0
                
                mdm2gg = 0
                mdm2gt = 0
                mdm2tt = 0
                
                GAL3aa = 0
                GAL3ca = 0
                GAL3cc = 0
                
                TIM1cc = 0
                TIM1gc = 0
                TIM1gg = 0
                
                if p16540 == "C/C":
                    p16540cc = 1
                elif p16540 == "G/C":
                    p16540gc = 1
                else:
                    p16540gg = 1
                
                if p16580 == "C/C":
                    p16580cc = 1
                elif p16580 == "C/T":
                    p16580ct = 1
                else:
                    p16580tt = 1
                
                if mdm2 == "G/G":
                    mdm2gg = 1
                elif mdm2 == "G/T":
                    mdm2gt = 1
                else:
                    mdm2tt = 1
                
                if GAL3 == "A/A":
                    GAL3aa = 1
                elif GAL3 == "C/A":
                    GAL3ca = 1
                else:
                    GAL3cc = 1
                    
                if TIM1 == "C/C":
                    TIM1cc = 1
                elif TIM1 == "G/C":
                    TIM1gc = 1
                else:
                    TIM1gg = 1
                print(age,p16540cc,p16540gc,p16540gg,p16580cc,p16580ct,p16580tt,mdm2gg,mdm2gt,mdm2tt,GAL3aa,GAL3ca,GAL3cc,TIM1cc,TIM1gc,TIM1gg)

                prd = loadedmodel.predict([[float(age),bool(p16540cc),bool(p16540gc),bool(p16540gg),bool(p16580cc),bool(p16580ct),bool(p16580tt),
                                            bool(mdm2gg),bool(mdm2gt),bool(mdm2tt),bool(GAL3aa),bool(GAL3ca),bool(GAL3cc),bool(TIM1cc),bool(TIM1gc),bool(TIM1gg)]])

                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Colorectal Cancer',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                try:
                    age = float(request.form['ageColorectal'])
                except ValueError:
                    return jsonify({'error': 'Invalid input for age'}), 400
                p16540= request.form['p16540']
                p16580 = request.form['p16580']
                mdm2 = request.form['mdm2']
                GAL3 = request.form['GAL3']
                TIM1 = request.form['TIM1']
                
                p16540cc = 0
                p16540gc = 0
                p16540gg = 0
                
                p16580cc = 0
                p16580ct = 0
                p16580tt = 0
                
                mdm2gg = 0
                mdm2gt = 0
                mdm2tt = 0
                
                GAL3aa = 0
                GAL3ca = 0
                GAL3cc = 0
                
                TIM1cc = 0
                TIM1gc = 0
                TIM1gg = 0
                
                if p16540 == "C/C":
                    p16540cc = 1
                elif p16540 == "G/C":
                    p16540gc = 1
                else:
                    p16540gg = 1
                
                if p16580 == "C/C":
                    p16580cc = 1
                elif p16580 == "C/T":
                    p16580ct = 1
                else:
                    p16580tt = 1
                
                if mdm2 == "G/G":
                    mdm2gg = 1
                elif mdm2 == "G/T":
                    mdm2gt = 1
                else:
                    mdm2tt = 1
                
                if GAL3 == "A/A":
                    GAL3aa = 1
                elif GAL3 == "C/A":
                    GAL3ca = 1
                else:
                    GAL3cc = 1
                    
                if TIM1 == "C/C":
                    TIM1cc = 1
                elif TIM1 == "G/C":
                    TIM1gc = 1
                else:
                    TIM1gg = 1
                    
                print(age,p16540cc,p16540gc,p16540gg,p16580cc,p16580ct,p16580tt,mdm2gg,mdm2gt,mdm2tt,GAL3aa,GAL3ca,GAL3cc,TIM1cc,TIM1gc,TIM1gg)
                prd = loadedmodel.predict([[float(age),bool(p16540cc),bool(p16540gc),bool(p16540gg),bool(p16580cc),bool(p16580ct),bool(p16580tt),
                                            bool(mdm2gg),bool(mdm2gt),bool(mdm2tt),bool(GAL3aa),bool(GAL3ca),bool(GAL3cc),bool(TIM1cc),bool(TIM1gc),bool(TIM1gg)]])
                if prd[0] == "1":
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Colorectal Cancer"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            try:
                age = float(request.form['ageColorectal'])
            except ValueError:
                return jsonify({'error': 'Invalid input for age'}), 400
            p16540= request.form['p16540']
            p16580 = request.form['p16580']
            mdm2 = request.form['mdm2']
            GAL3 = request.form['GAL3']
            TIM1 = request.form['TIM1']
            
            p16540cc =0
            p16540gc = 0
            p16540gg = 0
            
            p16580cc = 0
            p16580ct = 0
            p16580tt = 0
            
            mdm2gg = 0
            mdm2gt = 0
            mdm2tt = 0
            
            GAL3aa = 0
            GAL3ca = 0
            GAL3cc = 0
            
            TIM1cc = 0
            TIM1gc = 0
            TIM1gg = 0
            
            if p16540 == "C/C":
                p16540cc = 1
            elif p16540 == "G/C":
                p16540gc = 1
            else:
                p16540gg = 1
            
            if p16580 == "C/C":
                p16580cc = 1
            elif p16580 == "C/T":
                p16580ct = 1
            else:
                p16580tt = 1
            
            if mdm2 == "G/G":
                mdm2gg = 1
            elif mdm2 == "G/T":
                mdm2gt = 1
            else:
                mdm2tt = 1
            
            if GAL3 == "A/A":
                GAL3aa = 1
            elif GAL3 == "C/A":
                GAL3ca = 1
            else:
                GAL3cc = 1
                
            if TIM1 == "C/C":
                TIM1cc = 1
            elif TIM1 == "G/C":
                TIM1gc = 1
            else:
                TIM1gg = 1
                
            print(age,p16540cc,p16540gc,p16540gg,p16580cc,p16580ct,p16580tt,mdm2gg,mdm2gt,mdm2tt,GAL3aa,GAL3ca,GAL3cc,TIM1cc,TIM1gc,TIM1gg)
            prd = loadedmodel.predict([[float(age),bool(p16540cc),bool(p16540gc),bool(p16540gg),bool(p16580cc),bool(p16580ct),bool(p16580tt),
                                        bool(mdm2gg),bool(mdm2gt),bool(mdm2tt),bool(GAL3aa),bool(GAL3ca),bool(GAL3cc),bool(TIM1cc),bool(TIM1gc),bool(TIM1gg)]])
            if prd[0] == "1":
                result = "Positive"
            else:
                result = "Negative"
            disease = "Colorectal Cancer"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
    except Exception as e:
        print("An error occurred: ", str(e))
        return str(e), 500
    finally:
        client.close()

#Basem Team
@app.route('/Strokediagnosis', methods=['POST'])
def strokediagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Stroke", "Active": 1}
        result = db.model.find_one(query)
        print('we query')
        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)
        print('got the model stats')
        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # **check Path**
        print('loaded the model')
        gender = 1 if request.form['LS_Sex'] == 'Male' else 0
        age = int(request.form['LS_Age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart disease'])
        ever_married = int(request.form['ever married'])
        work_type = int(request.form['work type'])
        residence_type = int(request.form['residence type'])
        avg_glucose_level = int(request.form['avg glucose level'])
        bmi = int(request.form['bmi'])
        smoking_status = int(request.form['smoking status'])
        age_category = int(request.form['age category'])
        glucose_category = int(request.form['glucose category'])
        bmi_category = int(request.form['bmi category'])
        print('got the parameters')
        prd = loadedmodel.predict([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status, age_category, glucose_category, bmi_category]])
        
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"
        if 'role' in session:
            

            print(result)

            prediction = {
                'disease': 'Stroke',
                'prediction': result,
                'accuracy': accuracy
            }

            
            if session['role'] == 'laboratory specialist':
                

                return prediction
            else:
                
             
                disease = "Stroke"

                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            

            disease = "Stroke"

            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()



@app.route('/PC2diagnosis', methods=['POST'])
def pc2diagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Pancreatic Cancer", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # **check Path**
        age = int(request.form['LS_Age'])
        plasma = float(request.form['plasma'])
        creatinine = float(request.form['creatinine'])
        lyve1 =  float(request.form['lyve1'])
        reg1b = float(request.form['reg1b'])
        tff1 = float(request.form['tff1'])
        rega1a = float(request.form['rega1a'])
        sex_m = 1 if request.form['LS_Sex'] == 'Male' else 0
        sex_f = 1 if request.form['LS_Sex'] == 'Female' else 0
        prd = loadedmodel.predict([[age, plasma, creatinine, lyve1, reg1b, tff1, rega1a, sex_f, sex_m]])
            
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"

        if 'role' in session:
            

            print(result)

            prediction = {
                'disease': 'Pancreatic Cancer',
                'prediction': result,
                'accuracy': accuracy
            }

            
            if session['role'] == 'laboratory specialist':
                

                return prediction
            else:
                
             
                disease = "Pancreatic Cancer"

                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            

            disease = "Pancreatic Cancer"

            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()


@app.route('/PCOSdiagnosis', methods=['POST'])
def pcosdiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "PCOS", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)
        
        loadedmodel = pickle.load(open('./static/model/' + modelName, 'rb'))  # **check Path**
        age = int(request.form['LS_Age'])
        weight = float(request.form['Weight (Kg)'])
        height = float(request.form['Height(Cm)'])
        bmi = float(request.form['BMI'])
        blood_group = request.form['Blood Group']
        pulse_rate = float(request.form['Pulse rate(bpm)'])
        rr = float(request.form['RR (breaths/min)'])
        hb = float(request.form['Hb(g/dl)'])
        cycle_ri = int(request.form['Cycle(R/I)'] == 'R')
        cycle_length = int(request.form['Cycle length(days)'])
        marriage_status = int(request.form['Marraige Status (Yrs)'])
        pregnant = int(request.form['Pregnant(Y/N)'] == 'Y')
        num_aborptions = int(request.form['No. of aborptions'])
        beta_hcg_i = float(request.form['  I   beta-HCG(mIU/mL)'])
        beta_hcg_ii = float(request.form['II    beta-HCG(mIU/mL)'])
        fsh = float(request.form['FSH(mIU/mL)'])
        lh = float(request.form['LH(mIU/mL)'])
        fsh_lh = float(request.form['FSH/LH'])
        hip = float(request.form['Hip(inch)'])
        waist = float(request.form['Waist(inch)'])
        waist_hip_ratio = float(request.form['Waist:Hip Ratio'])
        tsh = float(request.form['TSH (mIU/L)'])
        amh = float(request.form['AMH(ng/mL)'])
        prl = float(request.form['PRL(ng/mL)'])
        vit_d3 = float(request.form['Vit D3 (ng/mL)'])
        prg = float(request.form['PRG(ng/mL)'])
        rbs = float(request.form['RBS(mg/dl)'])
        weight_gain = int(request.form['Weight gain(Y/N)'] == 'Y')
        hair_growth = int(request.form['hair growth(Y/N)'] == 'Y')
        skin_darkening = int(request.form['Skin darkening (Y/N)'] == 'Y')
        hair_loss = int(request.form['Hair loss(Y/N)'] == 'Y')
        pimples = int(request.form['Pimples(Y/N)'] == 'Y')
        fast_food = int(request.form['Fast food (Y/N)'] == 'Y')
        reg_exercise = int(request.form['Reg.Exercise(Y/N)'] == 'Y')
        bp_systolic = float(request.form['BP _Systolic (mmHg)'])
        bp_diastolic = float(request.form['BP _Diastolic (mmHg)'])
        follicle_no_l = int(request.form['Follicle No. (L)'])
        follicle_no_r = int(request.form['Follicle No. (R)'])
        avg_f_size_l = float(request.form['Avg. F size (L) (mm)'])
        avg_f_size_r = float(request.form['Avg. F size (R) (mm)'])
        endometrium = float(request.form['Endometrium (mm)'])
        prd = loadedmodel.predict([[age, weight, height, bmi, blood_group, pulse_rate, rr, hb, cycle_ri, cycle_length, marriage_status, pregnant, num_aborptions, beta_hcg_i, beta_hcg_ii, fsh, lh, fsh_lh, hip, waist, waist_hip_ratio, tsh, amh, prl, vit_d3, prg, rbs, weight_gain, hair_growth, skin_darkening, hair_loss, pimples, fast_food, reg_exercise, bp_systolic, bp_diastolic, follicle_no_l, follicle_no_r, avg_f_size_l, avg_f_size_r, endometrium]])
        print('prd:'+str(prd))
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"
        
        if 'role' in session:
            
            

            print(result)

            prediction = {
                'disease': 'PCOS',
                'prediction': result,
                'accuracy': accuracy
            }

            
            if session['role'] == 'laboratory specialist':
                

                return prediction
            else:
                
                
                disease = "PCOS"

                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            

            disease = "PCOS"
            
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        
    finally:
        client.close()

#Fai Group 

@app.route('/Osteodiagnosis', methods=['POST'])
def Osteodiagnosis():
    try:
        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Osteoporosis", "Active": 1}
    
        result = db.model.find_one(query)

        if result is None:
            print('no Active model')
        else:
            print('model found',result)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = joblib .load('./static/model/' + modelName)

        print('request data', request.data)
        print('request form',request.form)


        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                Joint_Pain = request.form['LS_Joint_Pain']
                sex = request.form['LS_Sex']
                age = request.form['LS_Age']
                height_in_meter = request.form['LS_height_in_meter']
                Weight_in_KG = request.form['LS_Weight_in_KG']
                Catsmoking = request.form['LS_Catsmoking']
                DiabetesAttr = request.form['LS_DiabetesAttr']
                Hypothyroidism = request.form['LS_Hypothyroidism']
                Seizure_Disorder = request.form['LS_Seizure_Disorder']
                Estrogen_Use = request.form['LS_Estrogen_Use']
                Dialysis = request.form['LS_Dialysis']
                Family_History_of_Osteo = request.form['LS_Family_History_of_Osteo']
                Maximum_Walking_distance_in_km = request.form['LS_Maximum_Walking_distance_in_km']
                Daily_Eating_habits = request.form['LS_Daily_Eating_habits']
                BMI = request.form['LS_BMI']
                Obesity = request.form['LS_Obesity']

                if sex == "Male":
                    sex = 1
                else:
                    sex = 0

                if Joint_Pain == 'Yes':
                    Joint_Pain = 1
                else:
                    Joint_Pain = 0

               

                if Hypothyroidism == 'Yes':
                    Hypothyroidism = 1
                else:
                    Hypothyroidism = 0

                if Seizure_Disorder == 'Yes':
                    Seizure_Disorder = 1
                else:
                    Seizure_Disorder = 0

                if Estrogen_Use == 'Yes':
                    Estrogen_Use = 1
                else:
                    Estrogen_Use = 0

                

                if Dialysis == 'Yes':
                    Dialysis = 1
                else:
                    Dialysis = 0

                if Family_History_of_Osteo == 'Yes':
                    Family_History_of_Osteo = 1
                else:
                    Family_History_of_Osteo = 0

                if Daily_Eating_habits == 'Normal':
                    Daily_Eating_habits = 1
                elif Daily_Eating_habits == 'low_macronutrients':
                    Daily_Eating_habits = 0

              
               

                if Obesity == 'normal_weight':
                    Obesity = 0
                elif Obesity == 'over_weight' :
                    Obesity = 2
                elif Obesity == 'obesity':
                    Obesity = 1
                elif Obesity =='under_weight':
                    Obesity =3
            

                if Catsmoking == 'Yes':
                    Catsmoking = 1
                else:
                    Catsmoking = 0

                if DiabetesAttr == 'Yes':
                    DiabetesAttr = 1
                else:
                    DiabetesAttr = 0

                prd = loadedmodel.predict([[float(Joint_Pain),float(sex), float(age), float(height_in_meter),
                                            float(Weight_in_KG), float(Catsmoking),float(DiabetesAttr), float(Hypothyroidism), float(Seizure_Disorder), float(Estrogen_Use),
                                            float(Dialysis), float(Family_History_of_Osteo), float(Maximum_Walking_distance_in_km), 
                                            float(Daily_Eating_habits),float(BMI), float(Obesity)]])

          
                
                if prd[0] == 1:
                    result = "osteopenia"
                elif prd[0] == 2:
                    result = "osteoporosis"
                else:
                    result = "Normal"
                

                prediction = {
                   'disease': 'Osteoporosis',
                   'prediction': result,
                    'accuracy': accuracy}
                return prediction
            
                
            
            else:
                Joint_Pain = request.form['Joint_PainOS']
                sex = request.form['SexOS']
                age = request.form['AgeOS']
                height_in_meter = request.form['height_in_meterOS']
                Weight_in_KG = request.form['Weight_in_KGOS']
                Catsmoking = request.form['CatSmokingOS']
                DiabetesAttr = request.form['DiabetesAttrOS']

                Hypothyroidism = request.form['HypothyroidismOS']
                Seizure_Disorder = request.form['Seizure_DisorderOS']
                Estrogen_Use = request.form['Estrogen_UseOS']
                Dialysis = request.form['DialysisOS']
                Family_History_of_Osteo = request.form['Family_History_of_OsteoOS']
                Maximum_Walking_distance_in_km = request.form['Maximum_Walking_distance_in_kmOS']
                Daily_Eating_habits = request.form['Daily_Eating_habitsOS']
                BMI = request.form['BMIOS']
                Obesity = request.form['ObesityOS']
                
                if sex == "Male":
                    sex = 1
                else:
                    sex = 0


                if Joint_Pain == 'Yes':
                    Joint_Pain = 1
                else:
                    Joint_Pain = 0

                

                if Hypothyroidism == 'Yes':
                    Hypothyroidism = 1
                else:
                    Hypothyroidism = 0

                if Seizure_Disorder == 'Yes':
                    Seizure_Disorder = 1
                else:
                    Seizure_Disorder = 0

                if Estrogen_Use == 'Yes':
                    Estrogen_Use = 1
                else:
                    Estrogen_Use = 0

                

                if Dialysis == 'Yes':
                    Dialysis = 1
                else:
                    Dialysis = 0

                if Family_History_of_Osteo == 'Yes':
                    Family_History_of_Osteo = 1
                else:
                    Family_History_of_Osteo = 0

                if Daily_Eating_habits == 'Normal':
                    Daily_Eating_habits = 1
                elif Daily_Eating_habits == 'low_macronutrients':
                    Daily_Eating_habits = 0


                if Obesity == 'normal_weight':
                    Obesity = 0
                elif Obesity == 'over_weight' :
                    Obesity = 2
                elif Obesity == 'obesity':
                    Obesity = 1
                elif Obesity =='under_weight':
                    Obesity =3
            

                if Catsmoking == 'Yes':
                    Catsmoking = 1
                else:
                    Catsmoking = 0

                if DiabetesAttr == 'Yes':
                    DiabetesAttr = 1
                else:
                    DiabetesAttr = 0

                prd = loadedmodel.predict([[float(Joint_Pain),float(sex), float(age), float(height_in_meter),
                                            float(Weight_in_KG), float(Catsmoking),float(DiabetesAttr), float(Hypothyroidism), float(Seizure_Disorder), float(Estrogen_Use),
                                            float(Dialysis), float(Family_History_of_Osteo), float(Maximum_Walking_distance_in_km), 
                                            float(Daily_Eating_habits),float(BMI), float(Obesity)]])
                
                
                
                if prd[0] == 1:
                    result = "osteopenia"
                elif prd[0] == 2:
                    result = "osteoporosis"
                else:
                    result = "Normal"
                
                disease = "Osteoporosis"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            Joint_Pain = request.form['Joint_PainOS']
            sex = request.form['SexOS']
            age = request.form['AgeOS']
            height_in_meter = request.form['height_in_meterOS']
            Weight_in_KG = request.form['Weight_in_KGOS']
            Catsmoking = request.form['CatSmokingOS']
            DiabetesAttr = request.form['DiabetesAttrOS']
            Hypothyroidism = request.form['HypothyroidismOS']
            Seizure_Disorder = request.form['Seizure_DisorderOS']
            Estrogen_Use = request.form['Estrogen_UseOS']
            Dialysis = request.form['DialysisOS']
            Family_History_of_Osteo = request.form['Family_History_of_OsteoOS']
            Maximum_Walking_distance_in_km = request.form['Maximum_Walking_distance_in_kmOS']
            Daily_Eating_habits = request.form['Daily_Eating_habitsOS']
            BMI = request.form['BMIOS']
            Obesity = request.form['ObesityOS']

            if sex == "Male":
                sex = 1
            else:
                sex = 0

            if Joint_Pain == 'Yes':
                    Joint_Pain = 1
            else:
                Joint_Pain = 0

            

            if Hypothyroidism == 'Yes':
                Hypothyroidism = 1
            else:
                Hypothyroidism = 0

            if Seizure_Disorder == 'Yes':
                Seizure_Disorder = 1
            else:
                Seizure_Disorder = 0

            if Estrogen_Use == 'Yes':
                Estrogen_Use = 1
            else:
                Estrogen_Use = 0

            

            if Dialysis == 'Yes':
                Dialysis = 1
            else:
                Dialysis = 0

            if Family_History_of_Osteo == 'Yes':
                Family_History_of_Osteo = 1
            else:
                Family_History_of_Osteo = 0

            if Daily_Eating_habits == 'Normal':
                Daily_Eating_habits = 1
            elif Daily_Eating_habits == 'low_macronutrients':
                Daily_Eating_habits = 0

            


            if Obesity == 'normal_weight':
                Obesity = 0
            elif Obesity == 'over_weight' :
                Obesity = 2
            elif Obesity == 'obesity':
                Obesity = 1
            elif Obesity =='under_weight':
                Obesity =3
        

            if Catsmoking == 'Yes':
                Catsmoking = 1
            else:
                Catsmoking = 0

            if DiabetesAttr == 'Yes':
                DiabetesAttr = 1
            else:
                DiabetesAttr = 0

            prd = loadedmodel.predict([[float(Joint_Pain),float(sex), float(age), float(height_in_meter),
                                            float(Weight_in_KG), float(Catsmoking),float(DiabetesAttr), float(Hypothyroidism), float(Seizure_Disorder), float(Estrogen_Use),
                                            float(Dialysis), float(Family_History_of_Osteo), float(Maximum_Walking_distance_in_km), 
                                            float(Daily_Eating_habits),float(BMI),float(Obesity)]])
            if prd[0] == 1:
                result = "osteopenia"
            elif prd[0] == 2:
                result = "osteoporosis"
            else:
                result = "Normal"

            disease = "Osteoporosis"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/SCAdiagnosis', methods=['POST'])
def SCAdiagnosis():
    try:

        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Sickle Cell Anemia", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = joblib .load('./static/model/' + modelName)
        





        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                
                hemoglobin = request.form['LS_Hemoglobin'] 
                PCV = request.form['LS_PCV']
                rbc_count = request.form['LS_RBCcount'] 
                mcv = request.form['LS_MCV'] 
                mchc = request.form['LS_MCHC']
             
            


                prd = loadedmodel.predict([[ float(hemoglobin), float(PCV), float(rbc_count), float(mcv),
                                            float(mchc)]])

                if prd[0] == 0:
                    result = "Negative"
                elif prd[0]==1:
                    result = "Positive"

                prediction = {
                    'disease': 'Sickle Cell Anemia',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            else:
                
                
                hemoglobin = request.form['HemoSCA']
                PCV = request.form['PcvSCA']
                rbc_count = request.form['RbcSCA']
                mcv = request.form['McvSCA']
                mchc = request.form['MchcSCA']
            

          
                

                prd = loadedmodel.predict([[float(hemoglobin), float(PCV), float(rbc_count), float(mcv),
                                                float(mchc)]])





                if prd[0] == 0:
                    result = "Negative"
                elif prd[0]==1:
                    result = "Positive"
                disease = "Sickle Cell Anemia"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            

            hemoglobin = request.form['HemoSCA']
            PCV = request.form['PcvSCA']
            rbc_count = request.form['RbcSCA']
            mcv = request.form['McvSCA']
            mchc = request.form['MchcSCA']
            
            
          
            prd = loadedmodel.predict([[float(hemoglobin), float(PCV), float(rbc_count), float(mcv),
                                            float(mchc)]])

            if prd[0] == 0:
                result = "Negative"
            elif prd[0]==1:
                result = "Positive"
                

            disease = "Sickle Cell Anemia"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)

    finally:
        client.close()

@app.route('/EpilepticSeizureDiagnosis', methods=['POST'])
def EpilepticSeizureDiagnosis(): 
    try:

        db, client = connection()

        # Query for the first model that matches the given criteria
        query = {"DiseaseType": "Epileptic Seizure", "Active": 1}
        result = db.model.find_one(query)

        # Extract the desired data from the result
        modleID = result["_id"]
        modelName = result["ModelName"]
        accuracy = result["Accuracy"]
        accuracy = float(accuracy)

        loadedmodel = joblib .load('./static/model/' + modelName)  # check Path

        if 'role' in session:
            if session['role'] == 'laboratory specialist':
                sex = request.form['LS_Sex']
                non_psychComorbidities = request.form['LS_non_psychComorbidities']
                PriorAEDs = request.form['LS_PriorAEDs']
                AsthmaAttr = request.form['LS_AsthmaAttr']
                Migraine = request.form['LS_Migraine']
                ChronicPain = request.form['LS_ChronicPain']
                DiabetesAttr = request.form['LS_DiabetesAttr']
                non_metastaticCancer = request.form['LS_non_metastaticCancer']
                NumberOfNoN_seizureNon_psychMedication = request.form['LS_NumberOfNoN_seizureNon_psychMedication']
                CurrentAEDs = request.form['LS_CurrentAEDs']
                Baseline_szFreq = request.form['LS_Baseline_szFreq']
                MedianDurationOfSeizures = request.form['LS_MedianDurationOfSeizures']
                NumberOfSeizureTypes = request.form['LS_NumberOfSeizureTypes']
                InjuryWithSeizure = request.form['LS_InjuryWithSeizure']
                Catamenial = request.form['LS_Catamenial']
                TriggerOfSleepDeprivation = request.form['LS_TriggerOfSleepDeprivation']
                Aura = request.form['LS_Aura']
                IctalEyeClosure = request.form['LS_IctalEyeClosure']
                IctalHallucinations = request.form['LS_IctalHallucinations']
                OralAutomatisms = request.form['LS_OralAutomatisms']
                Incontinence  = request.form['LS_Incontinence']
                LimbAutomatisms = request.form['LS_LimbAutomatisms']
                IctalTonic_clonic = request.form['LS_IctalTonic_clonic']
                MuscleTwitching = request.form['LS_MuscleTwitching']
                HipThrusting = request.form['LS_HipThrusting']
                Post_ictalFatigue = request.form['LS_Post_ictalFatigue']
                AnyHeadInjury = request.form['LS_AnyHeadInjury']
                PsychTraumaticEvents = request.form['LS_PsychTraumaticEvents']
                ConcussionWithoutLOC = request.form['LS_ConcussionWithoutLOC']
                ConcussionWithLOC = request.form['LS_ConcussionWithLOC']
                Severe_TBILOC = request.form['LS_Severe_TBILOC']
                Opioids = request.form['LS_Opioids']
                SexAbuse = request.form['LS_SexAbuse']
                PhysicalAbuse = request.form['LS_PhysicalAbuse']
                Rape  = request.form['LS_Rape']

                if sex == "Male":
                    sex = 1
                else:
                    sex = 0
               
                if AsthmaAttr == "Yes":
                    AsthmaAttr = 1
                else:
                    AsthmaAttr = 0

                if Migraine == "Yes":
                    Migraine = 1
                else:
                    Migraine = 0

                if ChronicPain == "Yes":
                    ChronicPain = 1
                else:
                    ChronicPain = 0

                if DiabetesAttr == "Yes":
                    DiabetesAttr = 1
                else:
                    DiabetesAttr = 0

                if non_metastaticCancer == "Yes":
                    non_metastaticCancer = 1
                else:
                    non_metastaticCancer = 0

                if InjuryWithSeizure == "Yes":
                    InjuryWithSeizure = 1
                else:
                    InjuryWithSeizure = 0

                
                if Catamenial == "Yes":
                    Catamenial = 1
                else:
                    Catamenial = 0
               
                if TriggerOfSleepDeprivation == "Yes":
                    TriggerOfSleepDeprivation = 1
                else:
                    TriggerOfSleepDeprivation = 0

                if Aura == "Yes":
                    Aura = 1
                else:
                    Aura = 0

                if IctalEyeClosure == "Yes":
                    IctalEyeClosure = 1
                else:
                    IctalEyeClosure = 0

                if IctalHallucinations == "Yes":
                    IctalHallucinations = 1
                else:
                    IctalHallucinations = 0

                if OralAutomatisms == "Yes":
                    OralAutomatisms = 1
                else:
                    OralAutomatisms = 0

                if Incontinence == "Yes":
                    Incontinence = 1
                else:
                    Incontinence = 0


                if LimbAutomatisms == "Yes":
                    LimbAutomatisms = 1
                else:
                    LimbAutomatisms = 0

                if IctalTonic_clonic == "Yes":
                    IctalTonic_clonic = 1
                else:
                    IctalTonic_clonic = 0

                
                if MuscleTwitching == "Yes":
                    MuscleTwitching = 1
                else:
                    MuscleTwitching = 0
               
                if HipThrusting == "Yes":
                    HipThrusting = 1
                else:
                    HipThrusting = 0

                if Post_ictalFatigue == "Yes":
                    Post_ictalFatigue = 1
                else:
                    Post_ictalFatigue = 0

                if AnyHeadInjury == "Yes":
                    AnyHeadInjury = 1
                else:
                    AnyHeadInjury = 0

                if PsychTraumaticEvents == "Yes":
                    PsychTraumaticEvents = 1
                else:
                    PsychTraumaticEvents = 0

                if ConcussionWithoutLOC == "Yes":
                    ConcussionWithoutLOC = 1
                else:
                    ConcussionWithoutLOC = 0

                if ConcussionWithLOC == "Yes":
                    ConcussionWithLOC = 1
                else:
                    ConcussionWithLOC = 0


                if Severe_TBILOC == "Yes":
                    Severe_TBILOC = 1
                else:
                    Severe_TBILOC = 0

                if Opioids == "Yes":
                    Opioids = 1
                else:
                    Opioids = 0

                if SexAbuse == "Yes":
                    SexAbuse = 1
                else:
                    SexAbuse = 0

                if PhysicalAbuse == "Yes":
                    PhysicalAbuse = 1
                else:
                    PhysicalAbuse = 0

                if Rape == "Yes":
                    Rape = 1
                else:
                    Rape = 0


                prd = loadedmodel.predict([[float(sex), float(non_psychComorbidities), float(PriorAEDs), float(AsthmaAttr),
                                            float(Migraine), float(ChronicPain), float(DiabetesAttr), float(non_metastaticCancer), float(NumberOfNoN_seizureNon_psychMedication),
                                            float(CurrentAEDs), float(Baseline_szFreq), float(MedianDurationOfSeizures), float(NumberOfSeizureTypes), float(InjuryWithSeizure),
                                            float(Catamenial), float(TriggerOfSleepDeprivation), float(Aura), float(IctalEyeClosure), float(IctalHallucinations),
                                            float(OralAutomatisms), float(Incontinence), float(LimbAutomatisms), float(IctalTonic_clonic), float(MuscleTwitching),
                                            float(HipThrusting), float(Post_ictalFatigue), float(AnyHeadInjury), float(PsychTraumaticEvents), float(ConcussionWithoutLOC),
                                            float(ConcussionWithLOC), float(Severe_TBILOC), float(Opioids), float(SexAbuse), float(PhysicalAbuse),
                                            float(Rape)]])



                if prd[0] == 0:
                    result = "Positive"
                else:
                    result = "Negative"

                prediction = {
                    'disease': 'Epileptic Seizure',
                    'prediction': result,
                    'accuracy': accuracy
                }
                return prediction
            
            else: 
                sex = request.form['SexES']
                non_psychComorbidities = request.form['non_psychComorbiditiesES']
                PriorAEDs = request.form['PriorAEDsES']
                AsthmaAttr = request.form['AsthmaAttrES']
                Migraine = request.form['MigraineES']
                ChronicPain = request.form['ChronicPainES']
                DiabetesAttr = request.form['DiabetesAttrES']
                non_metastaticCancer = request.form['non_metastaticCancerES']
                NumberOfNoN_seizureNon_psychMedication = request.form['NumberOfNoN_seizureNon_psychMedicationES']
                CurrentAEDs = request.form['CurrentAEDsES']
                Baseline_szFreq = request.form['Baseline_szFreqES']
                MedianDurationOfSeizures = request.form['MedianDurationOfSeizuresES']
                NumberOfSeizureTypes = request.form['NumberOfSeizureTypesES']
                InjuryWithSeizure = request.form['InjuryWithSeizureES']
                Catamenial = request.form['CatamenialES']
                TriggerOfSleepDeprivation = request.form['TriggerOfSleepDeprivationES']
                Aura = request.form['AuraES']
                IctalEyeClosure = request.form['IctalEyeClosureES']
                IctalHallucinations = request.form['IctalHallucinationsES']
                OralAutomatisms = request.form['OralAutomatismsES']
                Incontinence = request.form['IncontinenceES']
                LimbAutomatisms  = request.form['LimbAutomatismsES']
                IctalTonic_clonic = request.form['IctalTonic_clonicES']
                MuscleTwitching = request.form['MuscleTwitchingES']
                HipThrusting = request.form['HipThrustingES']
                Post_ictalFatigue = request.form['Post_ictalFatigueES']
                AnyHeadInjury = request.form['AnyHeadInjuryES']
                PsychTraumaticEvents = request.form['PsychTraumaticEventsES']
                ConcussionWithoutLOC = request.form['ConcussionWithoutLOCES']
                ConcussionWithLOC = request.form['ConcussionWithLOCES']
                Severe_TBILOC = request.form['Severe_TBILOCES']
                Opioids = request.form['OpioidsES']
                SexAbuse = request.form['SexAbuseES']
                PhysicalAbuse  = request.form['PhysicalAbuseES']
                Rape = request.form['RapeES']


                if sex == "Male":
                    sex = 1
                else:
                    sex = 0
                
                if AsthmaAttr == "Yes":
                    AsthmaAttr = 1
                else:
                    AsthmaAttr = 0

                if Migraine == "Yes":
                    Migraine = 1
                else:
                    Migraine = 0

                if ChronicPain == "Yes":
                    ChronicPain = 1
                else:
                    ChronicPain = 0

                if DiabetesAttr == "Yes":
                    DiabetesAttr = 1
                else:
                    DiabetesAttr = 0

                if non_metastaticCancer == "Yes":
                    non_metastaticCancer = 1
                else:
                    non_metastaticCancer = 0

                if InjuryWithSeizure == "Yes":
                    InjuryWithSeizure = 1
                else:
                    InjuryWithSeizure = 0

                
                if Catamenial == "Yes":
                    Catamenial = 1
                else:
                    Catamenial = 0
               
                if TriggerOfSleepDeprivation == "Yes":
                    TriggerOfSleepDeprivation = 1
                else:
                    TriggerOfSleepDeprivation = 0

                if Aura == "Yes":
                    Aura = 1
                else:
                    Aura = 0

                if IctalEyeClosure == "Yes":
                    IctalEyeClosure = 1
                else:
                    IctalEyeClosure = 0

                if IctalHallucinations == "Yes":
                    IctalHallucinations = 1
                else:
                    IctalHallucinations = 0

                if OralAutomatisms == "Yes":
                    OralAutomatisms = 1
                else:
                    OralAutomatisms = 0

                if Incontinence == "Yes":
                    Incontinence = 1
                else:
                    Incontinence = 0


                if LimbAutomatisms == "Yes":
                    LimbAutomatisms = 1
                else:
                    LimbAutomatisms = 0

                if IctalTonic_clonic == "Yes":
                    IctalTonic_clonic = 1
                else:
                    IctalTonic_clonic = 0

                
                if MuscleTwitching == "Yes":
                    MuscleTwitching = 1
                else:
                    MuscleTwitching = 0
               
                if HipThrusting == "Yes":
                    HipThrusting = 1
                else:
                    HipThrusting = 0

                if Post_ictalFatigue == "Yes":
                    Post_ictalFatigue = 1
                else:
                    Post_ictalFatigue = 0

                if AnyHeadInjury == "Yes":
                    AnyHeadInjury = 1
                else:
                    AnyHeadInjury = 0

                if PsychTraumaticEvents == "Yes":
                    PsychTraumaticEvents = 1
                else:
                    PsychTraumaticEvents = 0

                if ConcussionWithoutLOC == "Yes":
                    ConcussionWithoutLOC = 1
                else:
                    ConcussionWithoutLOC = 0

                if ConcussionWithLOC == "Yes":
                    ConcussionWithLOC = 1
                else:
                    ConcussionWithLOC = 0


                if Severe_TBILOC == "Yes":
                    Severe_TBILOC = 1
                else:
                    Severe_TBILOC = 0

                if Opioids == "Yes":
                    Opioids = 1
                else:
                    Opioids = 0

                if SexAbuse == "Yes":
                    SexAbuse = 1
                else:
                    SexAbuse = 0

                if PhysicalAbuse == "Yes":
                    PhysicalAbuse = 1
                else:
                    PhysicalAbuse = 0

                if Rape == "Yes":
                    Rape = 1
                else:
                    Rape = 0

                prd = loadedmodel.predict([[float(sex), float(non_psychComorbidities), float(PriorAEDs), float(AsthmaAttr),
                                            float(Migraine), float(ChronicPain), float(DiabetesAttr), float(non_metastaticCancer), float(NumberOfNoN_seizureNon_psychMedication),
                                            float(CurrentAEDs), float(Baseline_szFreq), float(MedianDurationOfSeizures), float(NumberOfSeizureTypes), float(InjuryWithSeizure),
                                            float(Catamenial), float(TriggerOfSleepDeprivation), float(Aura), float(IctalEyeClosure), float(IctalHallucinations),
                                            float(OralAutomatisms), float(Incontinence), float(LimbAutomatisms), float(IctalTonic_clonic), float(MuscleTwitching),
                                            float(HipThrusting), float(Post_ictalFatigue), float(AnyHeadInjury), float(PsychTraumaticEvents), float(ConcussionWithoutLOC),
                                            float(ConcussionWithLOC), float(Severe_TBILOC), float(Opioids), float(SexAbuse), float(PhysicalAbuse),
                                            float(Rape)]])
                if prd[0] == 0:
                    result = "Positive"
                else:
                    result = "Negative"
                disease = "Epileptic Seizure"
                return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
        else:
            sex = request.form['SexES']
            non_psychComorbidities = request.form['non_psychComorbiditiesES']
            PriorAEDs = request.form['PriorAEDsES']
            AsthmaAttr = request.form['AsthmaAttrES']
            Migraine = request.form['MigraineES']
            ChronicPain = request.form['ChronicPainES']
            DiabetesAttr = request.form['DiabetesAttrES']
            non_metastaticCancer = request.form['non_metastaticCancerES']
            NumberOfNoN_seizureNon_psychMedication = request.form['NumberOfNoN_seizureNon_psychMedicationES']
            CurrentAEDs = request.form['CurrentAEDsES']
            Baseline_szFreq = request.form['Baseline_szFreqES']
            MedianDurationOfSeizures = request.form['MedianDurationOfSeizuresES']
            NumberOfSeizureTypes = request.form['NumberOfSeizureTypesES']
            InjuryWithSeizure = request.form['InjuryWithSeizureES']
            Catamenial = request.form['CatamenialES']
            TriggerOfSleepDeprivation = request.form['TriggerOfSleepDeprivationES']
            Aura = request.form['AuraES']
            IctalEyeClosure = request.form['IctalEyeClosureES']
            IctalHallucinations = request.form['IctalHallucinationsES']
            OralAutomatisms = request.form['OralAutomatismsES']
            Incontinence = request.form['IncontinenceES']
            LimbAutomatisms  = request.form['LimbAutomatismsES']
            IctalTonic_clonic = request.form['IctalTonic_clonicES']
            MuscleTwitching = request.form['MuscleTwitchingES']
            HipThrusting = request.form['HipThrustingES']
            Post_ictalFatigue = request.form['Post_ictalFatigueES']
            AnyHeadInjury = request.form['AnyHeadInjuryES']
            PsychTraumaticEvents = request.form['PsychTraumaticEventsES']
            ConcussionWithoutLOC = request.form['ConcussionWithoutLOCES']
            ConcussionWithLOC = request.form['ConcussionWithLOCES']
            Severe_TBILOC = request.form['Severe_TBILOCES']
            Opioids = request.form['OpioidsES']
            SexAbuse = request.form['SexAbuseES']
            PhysicalAbuse  = request.form['PhysicalAbuseES']
            Rape = request.form['RapeES']

            if sex == "Male":
                sex = 1
            else:
                sex = 0
            
            if AsthmaAttr == "Yes":
                    AsthmaAttr = 1
            else:
                AsthmaAttr = 0

            if Migraine == "Yes":
                Migraine = 1
            else:
                Migraine = 0

            if ChronicPain == "Yes":
                ChronicPain = 1
            else:
                ChronicPain = 0

            if DiabetesAttr == "Yes":
                DiabetesAttr = 1
            else:
                DiabetesAttr = 0

            if non_metastaticCancer == "Yes":
                non_metastaticCancer = 1
            else:
                non_metastaticCancer = 0

            if InjuryWithSeizure == "Yes":
                InjuryWithSeizure = 1
            else:
                InjuryWithSeizure = 0

            
            if Catamenial == "Yes":
                Catamenial = 1
            else:
                Catamenial = 0
            
            if TriggerOfSleepDeprivation == "Yes":
                TriggerOfSleepDeprivation = 1
            else:
                TriggerOfSleepDeprivation = 0

            if Aura == "Yes":
                Aura = 1
            else:
                Aura = 0

            if IctalEyeClosure == "Yes":
                IctalEyeClosure = 1
            else:
                IctalEyeClosure = 0

            if IctalHallucinations == "Yes":
                IctalHallucinations = 1
            else:
                IctalHallucinations = 0

            if OralAutomatisms == "Yes":
                OralAutomatisms = 1
            else:
                OralAutomatisms = 0

            if Incontinence == "Yes":
                Incontinence = 1
            else:
                Incontinence = 0


            if LimbAutomatisms == "Yes":
                LimbAutomatisms = 1
            else:
                LimbAutomatisms = 0

            if IctalTonic_clonic == "Yes":
                IctalTonic_clonic = 1
            else:
                IctalTonic_clonic = 0

            
            if MuscleTwitching == "Yes":
                MuscleTwitching = 1
            else:
                MuscleTwitching = 0
            
            if HipThrusting == "Yes":
                HipThrusting = 1
            else:
                HipThrusting = 0

            if Post_ictalFatigue == "Yes":
                Post_ictalFatigue = 1
            else:
                Post_ictalFatigue = 0

            if AnyHeadInjury == "Yes":
                AnyHeadInjury = 1
            else:
                AnyHeadInjury = 0

            if PsychTraumaticEvents == "Yes":
                PsychTraumaticEvents = 1
            else:
                PsychTraumaticEvents = 0

            if ConcussionWithoutLOC == "Yes":
                ConcussionWithoutLOC = 1
            else:
                ConcussionWithoutLOC = 0

            if ConcussionWithLOC == "Yes":
                ConcussionWithLOC = 1
            else:
                ConcussionWithLOC = 0


            if Severe_TBILOC == "Yes":
                Severe_TBILOC = 1
            else:
                Severe_TBILOC = 0

            if Opioids == "Yes":
                Opioids = 1
            else:
                Opioids = 0

            if SexAbuse == "Yes":
                SexAbuse = 1
            else:
                SexAbuse = 0

            if PhysicalAbuse == "Yes":
                PhysicalAbuse = 1
            else:
                PhysicalAbuse = 0

            if Rape == "Yes":
                Rape = 1
            else:
                Rape = 0

            prd = loadedmodel.predict([[float(sex), float(non_psychComorbidities), float(PriorAEDs), float(AsthmaAttr),
                                            float(Migraine), float(ChronicPain), float(DiabetesAttr), float(non_metastaticCancer), float(NumberOfNoN_seizureNon_psychMedication),
                                            float(CurrentAEDs), float(Baseline_szFreq), float(MedianDurationOfSeizures), float(NumberOfSeizureTypes), float(InjuryWithSeizure),
                                            float(Catamenial), float(TriggerOfSleepDeprivation), float(Aura), float(IctalEyeClosure), float(IctalHallucinations),
                                            float(OralAutomatisms), float(Incontinence), float(LimbAutomatisms), float(IctalTonic_clonic), float(MuscleTwitching),
                                            float(HipThrusting), float(Post_ictalFatigue), float(AnyHeadInjury), float(PsychTraumaticEvents), float(ConcussionWithoutLOC),
                                            float(ConcussionWithLOC), float(Severe_TBILOC), float(Opioids), float(SexAbuse), float(PhysicalAbuse),
                                            float(Rape)]])

       


            if prd[0] == 0:
                result = "Positive"
            else:
                result = "Negative"
            disease = "Epileptic Seizure"
            return render_template('diagnoseRR.html', result=result, disease=disease, acc=accuracy)
    finally:
        client.close()
# Fai's Group Dnnn -------------------------------------


# RANA & MUNERA VERSION
@app.route('/deleteaccount')
@login_required
def delUser():
    if session['role'] == 'registered user':
        try:
            db, client = connection()
            uname = session['username']

            dbacc = db.account.find_one({'_id': uname}, {'Name': 1, 'Email': 1})
            mrn = db.patient.find_one({'username': uname}, {'_id': 1})
            db.test.delete_many({'mrn': mrn['_id']})
            db.patient.delete_many({'_id': mrn['_id']})
            db.account.delete_many({'_id': uname})

            FLname = dbacc['Name'].split(" ")
            # render the Deleteuser_email html template with the user first name
            html = render_template('Deleteuser_email.html', user=FLname[0])
            # send the email to user with the Deleteuser_email html template
            msg = email("Account Deleted", dbacc['Email'], html)
            mail.send(msg)
            flash("Account successfully deleted.", "info")
            logingout()
        finally:
            # close the Database connection
            client.close()

        return render_template("Emailconfirm.html", title="Account Deleted")
    else:
        return render_template("404.html")

@app.route('/confirm/<token>')  # this route handles email address confirmation tokens requests
def confirm_email(token):
    try:
        # extract the email address from the token 
        confirm_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        email = confirm_serializer.loads(token, salt='email-confirmation-salt', max_age=86400)
    except:  # check if the token is invalde or has expired
        flash('The confirmation link is invalid or has expired.', 'info')
        # render Emailconfirm html the will disply the token info message 
        return render_template("Emailconfirm.html", title="Email Confirmation")
    try:
        # get the Database connection
        db, client = connection()
        # fetch the username and EmailConfirmed flag vaule of the emial confirming user account
        data = db.account.find_one({'Email': email})
        uname = data['_id']
        econfirmed = data['EmailConfirmed']
        if econfirmed == 1:  # check if the email has been confrimed
            flash('Account already confirmed. Please login.', 'info')
        else:
            # update emial confirming user account by setting the EmailConfirmed to 1 'confirmed' and ConfirmedEmailOn with with current timestamp
            db.account.find_one_and_update({'_id': uname}, {'$set': {'EmailConfirmed': 1}})
            flash('Thank you for confirming your email address!', 'info')
        # render Emailconfirm html the will disply the token info message
    finally:
        # close the Database connection
        client.close()

    return render_template("Emailconfirm.html", title="Email Confirmation")

@app.route('/video')
def video():
    return render_template("video.html")

def validate_pass(form, field):
    letter_flag = False
    number_flag = False
    for i in field.data:
        if i.isalpha():
            letter_flag = True
        if i.isdigit():
            number_flag = True
    if not (letter_flag) or not (number_flag):
        raise ValidationError('Password should be at least 8 alphanumeric characters.')

def validate_name(form, field):
    number_flag = False
    for i in field.data:
        if i.isdigit():
            number_flag = True
    if number_flag:
        raise ValidationError('Numbers are not allowed in name.')

def validate_date(form, field):
    bdate = field.data
    today = date.today()
    years_ago = date.today() - relativedelta(years=115)
    if bdate >= today or years_ago >= bdate:
        raise ValidationError('Invalid date for date of birth.')

class Register(Form):
    name = StringField('Name', [DataRequired(), validate_name, validators.length(min=3, max=40)])
    ID = StringField('Username', [DataRequired()])  # , is_username_available #validators.length(min=10, max=10,
    uemail = StringField('Email', [DataRequired(), validators.Email()])
    gender = SelectField(choices=[('', ''), ('Female', 'Female'), ('Male', 'Male')], validators=[DataRequired()])
    birthdate = DateField('Date of birth', [validators.DataRequired(), validate_date], format="%Y-%m-%d")
    password = PasswordField('Password', [validators.DataRequired(), validators.length(min=8,
                                                                                message="Password should be at least 8 alphanumeric characters."),
                                        validate_pass,
                                        validators.EqualTo('cpassword', message='Passwords do not match.')])
    cpassword = PasswordField('Confirm Password')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = Register(request.form)
    if request.method == 'GET':
        return render_template('register.html', form=form)

    elif request.method == 'POST' and form.validate():
        name = form.name.data
        username = form.ID.data.lower()
        uemail = form.uemail.data.lower()
        gender = form.gender.data
        birthdate = str(form.birthdate.data)
        password = sha256_crypt.encrypt(str(form.password.data))

        db, client = connection()

        data = db.account.find_one({'_id': username})
        dataem = db.account.find_one({'Email': uemail})

        if data is not None:
            error = "The Username is already taken"
            return render_template('register.html', form=form, error=error)

        elif dataem is not None:
            error = "Email already exists in another account"
            return render_template('register.html', form=form, error=error)
        else:
            role = "registered user"

            data = db.patient.find_one({'username': username.lower()})
            if data is not None:
                db.patients.updateOne(
                    {'username': username.lower()},
                    {'$set': {'Name': name, 'birth_date': birthdate, 'gender': gender}})
            else:
                mrn = random.randint(1000000, 999999999)

                while db.patient.count_documents({'_id': mrn}) > 0:
                    mrn = random.randint(1000000, 999999999)

                now = datetime.now()
                midnight = datetime(now.year, now.month, now.day)
                d1 = {'_id': username, 'Name': name, 'Email': uemail, 'TempPassFlag': '0', 'Password': password,
                      'Role': role, 'EmailConfirmed': str(midnight)}
                db.account.insert_one(d1)

                d2 = {'_id': mrn, 'username': username, 'birth_date': birthdate, 'gender': gender}
                db.patient.insert_one(d2)

            send_confirmation_email(uemail, name, 'createaccount_email.html')
            session['logged_in'] = True
            session['username'] = username
            session['role'] = role
            return redirect(url_for('profile'))

    elif request.method == 'POST':
        error = "All field must be filled"
        return render_template('register.html', form=form, error=error)

@app.route('/profile')
@login_required  # this route handles profile disply requests
def profile():
    try:
        uname = session['username'].lower()

        db, client = connection()
        q = db.account.find_one({'_id': uname}, {'Name': 1, 'Email': 1, 'Role': 1, 'TempPassFlag': 1})
        dbname = q['Name']
        dbemail = q['Email'].lower()
        dbrole = q['Role']
        dbtemppassflg = q['TempPassFlag']

        if dbrole == 'medical specialist':
            page = 'MSprofile.html'

        elif dbrole == 'admin':
            page = 'admin.html'

        elif dbrole == 'laboratory specialist':
            page = 'LSprofile.html'

        elif dbrole == 'registered user':
            page = 'RUprofile.html'
            q = db.patient.find_one({'username': uname}, {'gender': 1, 'birth_date': 1})
            dbgender = str(q['gender'])
            dbbdate = q['birth_date']

    finally:
        client.close()

    if dbrole == 'registered user':  # check if the user is a registered user
        if 'passerror' in request.args:  # check if a password  update error exist
            # get the password update error flag value
            passerror = request.args.get('passerror')
            # render the user profile template with the profile info and password update error flag
            return render_template(page, passerror=passerror, name=dbname, email=dbemail, gender=dbgender,
                                   bdate=dbbdate)
        elif 'emailerror' in request.args:  # check if a email updete error exist
            # get the email error flag value
            emailerror = request.args.get('emailerror')
            # get the entered old and new email values
            if 'olde' in request.args:
                olde = request.args.get('olde')
                newe = request.args.get('newe')
                # render the user profile template with the profile info and the email from old, email update error flag and new email inputted values
                return render_template(page, emailerror=emailerror, name=dbname, email=dbemail, gender=dbgender,
                                       bdate=dbbdate, oldemail=olde, newemail=newe)
            # render the user profile template with the profile info and the email from old, email update error flag and new email inputted values
            # render the user profile template with the profile info and the email update error flag
            return render_template(page, emailerror=emailerror, name=dbname, email=dbemail, gender=dbgender,
                                   bdate=dbbdate)
        elif dbtemppassflg == 1:
            flash("Please update your temporary password", 'error')
            return render_template(page, passerror="True", name=dbname, email=dbemail, gender=dbgender, bdate=dbbdate)
        else:
            # render the user profile template with the profile info only
            return render_template(page, name=dbname, email=dbemail, gender=dbgender, bdate=dbbdate)
    else:

        if 'passerror' in request.args:
            # get the password update error flag value
            passerror = request.args.get('passerror')
            # render the user profile template with the profile info and password update error flag
            return render_template(page, passerror=passerror, name=dbname, email=dbemail)

        elif 'emailerror' in request.args:
            # get the email error flag value
            emailerror = request.args.get('emailerror')
            # get the entered old and new email values
            if 'olde' in request.args:
                olde = request.args.get('olde')
                newe = request.args.get('newe')
                # render the user profile template with the profile info and the email from old, email update error flag and new email inputted values
                return render_template(page, emailerror=emailerror, name=dbname, email=dbemail, oldemail=olde,
                                       newemail=newe)
            # render the user profile template with the profile info and the email from  update error flag
            return render_template(page, emailerror=emailerror, name=dbname, email=dbemail)
        elif dbtemppassflg == '1':
            flash("Please update your temporary password", 'error')
            return render_template(page, passerror="True", name=dbname, email=dbemail)
        else:
            # render the user profile template with the profile info only
            return render_template(page, name=dbname, email=dbemail)

@app.route('/passwordchange', methods=['POST'])  # this route handles password change requests
def changepassword():
    uname = session['username']

    frmoldpass = request.form['oldpassword']
    frmnewpass = request.form['newpassword']
    try:
        # get the Database connection
        db, client = connection()
        q = db.account.find_one({'_id': uname}, {'Password': 1, 'Email': 1, 'Name': 1})

        if not (sha256_crypt.verify(frmoldpass, q[
            'Password'])):  # check if the old password match the hashed password of the user account
            flash("Old password is incorrect", 'error')
        elif (frmoldpass == frmnewpass):  # check if the new password match the old password of the user account
            flash("Old password matches the new password", 'error')
        else:  # if not any of the above then update user accout password with the hashed new password
            # hash the new password
            hashedpassword = sha256_crypt.hash(frmnewpass)
            db.account.find_one_and_update({'_id': uname}, {'$set': {'Password': hashedpassword, 'TempPassFlag': 0}})

            # split the user name to first and last name
            FLname = q['Name'].split(" ")
            # render the change_password_email html template with the user first name
            html = render_template('change_password_email.html', user=FLname[0])
            # send the email to user with the change_password_email html template
            msg = email("Changed password", q['Email'], html)
            mail.send(msg)
            flash("Password is successfully changed.", 'info')
            # return with no errors and update confirmation message
            return redirect(url_for('profile', passerror="False"))

    finally:
        client.close()

    # return with the error, the error message and the entered values 
    return redirect(url_for('profile', passerror="True"))

@app.route('/emailchange', methods=['POST'])  # this route handles email change requests
def changeemail():
    uname = session['username']

    frmoldemail = request.form['oldemail'].lower()
    frmnewemail = request.form['newemail'].lower()
    try:
        # get the Database connection
        db, client = connection()
        # fetch the email and name of the logged in user account
        q1 = db.account.find_one({'_id': uname}, {'Email': 1, 'Name': 1})
        # check if the new email is exist in anthor user account

        if frmoldemail != q1['Email']:  # check if the old email match the email of the user account
            flash("Old email is incorrect", 'error')
        elif frmoldemail == frmnewemail:  # check if the new email match the old email of the user account
            flash("New email matches the old email", 'error')
        elif db.account.count_documents(
                ({'Email': frmnewemail})) > 0:  # check if the new email is exist in anthor user account
            flash("New email already exists in another account", 'error')
        else:  # if not any of the above then update email but keep it unvalidated and send an emial confirmation email to the user new email also set the EmailConfirmationSentOn with current timestamp
            db.account.find_one_and_update({'_id': uname}, {
                '$set': {'Email': frmnewemail, 'EmailConfirmationSentOn': str(datetime.now()), 'EmailConfirmed': 0}})

            # send an email confirmation email to the user new email
            send_confirmation_email(frmnewemail, q1['Name'], 'change_email_confirmation.html')
            flash("Email updated, please confirm your new email address (link sent to new email).", 'info')
            # return with no errors and update confirmation message
            return redirect(url_for('profile', emailerror="False"))
    finally:
        client.close()
        # close the Database connection
    # return with the error, the error message and the entered values 
    return redirect(url_for('profile', emailerror="True", olde=frmoldemail, newe=frmnewemail))

@app.route('/profileupdate', methods=['POST'])  # this route handles the registered user profie update requests
def updeteprofile():
    uname = session['username']
    # get the user profile info from the profile form
    frmname = request.form['profile-name']
    frmgender = request.form['profile-gender']
    frmbdate = request.form['birth-date']
    try:
        # get the Database connection
        db, client = connection()
        # update the user patient record with new profile info name, gender and birth date
        db.patient.find_one_and_update({'username': uname},
                                       {'$set': {'gender': frmgender, 'birth_date': frmbdate}})

        db.account.find_one_and_update({'_id': uname}, {'$set': {'Name': frmname}})

        flash("Profile successfully updated.", 'info')
    finally:
        client.close()

    # return with update confirmation message
    return redirect(url_for('profile'))

@app.route('/manageUser')
@login_required
def displayUsers():
    if session['role'] == 'admin':
        try:
            db, client = connection()
            # fetch user data from MongoDB
            users = db.account.find({}, {'_id': 1, 'Name': 1, 'Role': 1, 'Email': 1})
            usersList = []
            for user in users:
                if 'Name' in user:
                    if user.get('Role') != 'registered user':
                        usersList.append((user['_id'], user['Name'], user.get('Role'), user.get('Email')))
        finally:
            # close the MongoDB connection
            client.close()

        if 'removeerror' in request.args:
            removeerror = request.args.get('removeerror')
            return render_template('adminManageUser.html', usersList=usersList, registeredName="Khawla",
                                   removeerror="False")
        if 'adderror' in request.args:
            adderror = request.args.get('adderror')
            if 'name' in request.args:
                name = request.args.get('name')
                email = request.args.get('email')
                role = request.args.get('role')
                return render_template('adminManageUser.html', usersList=usersList, registeredName="Khawla",
                                       adderror="True", Name=name, Email=email, Role=role)
            return render_template('adminManageUser.html', usersList=usersList, registeredName="Khawla",
                                   adderror="False")
        return render_template('adminManageUser.html', usersList=usersList, registeredName="Khawla")
    else:
        return render_template("404.html")

@app.route('/removeUser', methods=['POST'])
def removeUser():
    try:
        db, client = connection()
        users = request.form.getlist("Users")

        for user_id in users:
            user_data = db.account.find_one({'_id': user_id}, {'Name': 1, 'Email': 1})
            dbname = user_data['Name']
            dbemail = user_data['Email']

            db.account.delete_one({'_id': user_id})

            FLname = dbname.split(" ")
            # render the Deleteuser_email html template with the user first name
            html = render_template('Deleteuser_email.html', user=FLname[0])
            # send the email to user with the Deleteuser_email html template
            msg = email("Account Deleted", dbemail, html)
            mail.send(msg)
    finally:
        client.close()

    flash("User(s) successfully removed", 'info')
    return redirect(url_for('displayUsers', removeerror="False"))

# Name - Email and use it as username
@app.route('/addUser', methods=['POST'])
def addUser():
    try:
        db, client = connection()

        newName = request.form['inputName']
        newEmail = str(request.form['inputEmail']).lower()
        newRole = request.form.get('roleSelect')
        tempPass = password_gen()
        hashedTempPass = sha256_crypt.hash(tempPass)

        username = newEmail.split('@')[0]

        dataem = db.account.find_one({'Email': newEmail})
    finally:
        client.close()

    if dataem is not None:
        flash("The email already exists in another account", 'error')
        return redirect(url_for('displayUsers', adderror="True", name=newName, email=newEmail, role=newRole))
    else:
        try:
            db, client = connection()

            exist = db.account.find_one({'_id': username})
            if exist is not None:
                username = username_gen(username.lower())

            # Create a new user account in the database
            user = {"_id": username, "Name": newName, "Email": newEmail, "TempPassFlag": 1, "Password": hashedTempPass,
                    "Role": newRole, "EmailConfirmed": 0} # 0 or 1
            db.account.insert_one(user)

            FLname = newName.split(" ")
            # render the Deleteuser_email html template with the user first name
            html = render_template('adduser_email.html', user=FLname[0], username=username, password=tempPass)
            # send the email to user with the Deleteuser_email html template
            msg = email("Account Created", newEmail, html)
            mail.send(msg)
            flash("User successfully added.", 'info')
        finally:
            client.close()

        return redirect(url_for('displayUsers', adderror="False"))

def password_gen():
    alphabet = string.ascii_letters
    digits = string.digits
    passwordalphabet = ''.join([choice(alphabet) for _ in range(5)])
    passworddigits = ''.join([choice(digits) for _ in range(3)])
    unshuffledpassword = passwordalphabet + passworddigits
    l = list(unshuffledpassword)
    random.shuffle(l)
    shuffledpassword = ''.join(l)
    return shuffledpassword

def username_gen(name):
    try:
        FLname = name.split(" ")
        username = FLname[0]
        if len(FLname) > 1:
            username += FLname[1]

        db, client = connection()

        existing_users = db.account.find({'_id': {'$regex': '^' + username}})
        existing_usernames = [user['_id'] for user in existing_users]

        for i in range(1001):
            if i == 0:
                new_username = username
            else:
                new_username = username + str(i)

            if new_username not in existing_usernames:
                User = new_username
                break
        return User
    finally:
        client.close()
