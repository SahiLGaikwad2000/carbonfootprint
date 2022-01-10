from flask import *
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import RMSprop
import os
import sys
# import os
import matplotlib.pyplot as plt
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


import numpy as npy
from util import base64_to_pil

import pyrebase
config = {
    "apiKey": "AIzaSyC0F7JrEgSKnbdOQjvmRXsNfDIxEAwLFI8",
    "authDomain": "test-4b021.firebaseapp.com",
    "databaseURL": "https://test-4b021-default-rtdb.firebaseio.com",
    "projectId": "test-4b021",
    "storageBucket": "test-4b021.appspot.com",
    "messagingSenderId": "417246562047",
    "appId": "1:417246562047:web:682a8de854f311d5962eca",
    "measurementId": "G-QDCCG49RQR"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
# db.child("names").push({"name": "rohan"})
# users = db.child("names").get()
# print(users.val())
# a=list(users.val())
# print(a)
# print(users.val().values())
# for i in users.val().values():
#     print(i)
# for i, j in users.val().items():
#     print(i, j)
#     if j['name'] == "sahil":
#         print("hello")
app = Flask(__name__)

# MODEL_PATH = 'F:\\animal prediction\\model\\mymodel.h5'
MODEL_PATH = 'model\\mymodel.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])


def model_predict(img, model):

    # dir_path="F:\\animal prediction\\uploads"
    dir_path = "uploads"

    for i in os.listdir(dir_path):
        img = image.load_img(dir_path+'\\'+i, target_size=(200, 200, 3))
        x = image.img_to_array(img)
        x = npy.expand_dims(x, axis=0)
        images = npy.vstack([x])
        val = model.predict(images)
        print(val)
        if (int(val[0][0])) == 1:
            # return ("Apples Emission: 0.331 KgCo2 per Kg")
            return ("Eggs\n Emission: 0.588 KgCo2 per Kg")
        if (int(val[0][1])) == 1:
            # return ("Banana Emission: 0.072 KgCo2 per Kg")
            return ("Apples Emission: 0.331 KgCo2 per Kg")
        if (int(val[0][2])) == 1:
            # return ("Eggs\n Emission: 0.588 KgCo2 per Kg")
            return ("Banana Emission: 0.072 KgCo2 per Kg")
        if (int(val[0][3])) == 1:
            return ("Meat Emission: 0.846 KgCo2 per Kg")
        if (int(val[0][4])) == 1:
            return ("Potato Emission: 0.025 KgCo2 per Kg")
        if (int(val[0][5])) == 1:
            return ("Rice Emission: 1.22 KgCo2 per Kg")


user = ''


@app.route('/', methods=['GET', 'POST'])
def basic():
    c = 0
    if request.method == 'POST':
        name = request.form['name']
        passwd = request.form['pass']
        users = db.child("names").get()
        print(users.val())
        if users.val():

            for i in users.val().values():
                print(i)
                if i['name'] == name or (i['name'] == name and i['pass'] == passwd):

                    print("already exist")
                    c = c+1
                    return render_template('register.html', co=c)
                    break
            if c == 0:
                db.child('names').push({"name": name, "pass": passwd})
                return redirect(url_for('checklog'))
        else:
            db.child('names').push({"name": name, "pass": passwd})
            return redirect(url_for('checklog'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def checklog():
    return render_template('login.html')


@app.route('/check', methods=['GET', 'POST'])
def afterlog():
    global user
    if request.method == 'POST':
        name = request.form['lname']
        passwd = request.form['lpass']
        users = db.child("names").get()
        print(users.val())
        if users.val():

            for i in users.val().values():
                print(i)
                if i['name'] == name and i['pass'] == passwd:
                    user = name
                    print("login success")

    return render_template('test.html')


a = 0


@app.route('/homepage', methods=['GET', 'POST'])
def check():
    global a
    if request.method == 'POST':
        b = request.form['but']
        print(b)
        a = a+1
        # print(a)

    return render_template('homepage.html', val=a)


@app.route('/google-charts/pie-chart')
def google_pie_chart():
    data = {'Task': 'Hours per Day', 'Work': 11, 'Eat': 2,
            'Commute': 2, 'Watching TV': 2, 'Sleeping': 7}
    # print(data)
    return render_template('pie.html', data=data)


a = 0
tcar = ''
tbike = ''
z = ''
eb = 0
lpg = 0
png = 0
oil = 0
water = 0
area = 0
np = 0


@app.route('/home', methods=['GET', 'POST'])
def homec():
    global a, np, area, water, lpg, png, eb, oil, user
    global tcar
    global tbike
    global z
    if request.method == 'POST':
        if a == 0:
            z = request.form['service']
            tcar = request.form['tcar']
            tbike = request.form['tbike']

            # print(z)
            a += 1
            return render_template('residential.html', val=z)
        if a == 1:
            eb = request.form['eb']
            if z == 'LPG (Cylinder)':
                lpg = request.form['lpg']
            else:
                png = request.form['png']
            oil = request.form['oil']
            water = request.form['water']
            area = request.form['area']
            np = request.form['np']

            a -= 1
            print(tcar)
            print(tbike)

            return render_template('transport.html', car=tcar, bike=tbike)
    return render_template('homec.html', user=user)


# def checkhomec():
#     if request.method == 'POST':
#         z = request.form['fw']
#         print(z)
#     return render_template('homec.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    global a, np, area, water, lpg, png, eb, oil
    global tcar
    global tbike
    global z
    trans = 0
    transcar = 0
    transbike = 0
    print(eb)

    ########## Residential ##########
    # reseb = ((int(eb)/8.8)*1.19)/(int(np))
    reseb = ((int(eb)/8.8)*0.385)/(int(np))
    reslpg = (int(lpg)*14.2*2.983)/(12*int(np))
    # respng=(png/13.6)*ef
    reswat = int(water)*0.000298*30
    resarea = ((int(area)/int(np))*20*0.9)/(40*12)

    ########## Transport ##########
    if tcar == 'Yes':
        trfw = request.form['fw']
        trac = request.form['ac']
        transcar = (int(trac)/100)*18*0.18
    if tbike == 'Yes':
        trtw = request.form['tw']
        trab = request.form['ab']
        transbike = (int(trab)/100)*40*0.3
    trtr = request.form['transport']
    if trtr == 'Private car/bike':
        pass
    elif trtr == 'Bus':
        trans = 0.822*10*30/9.2
    elif trtr == 'Train':
        trans = 0.041*10*30/(4000*24*30)

    elif trtr == 'Auto/Taxi':
        trans = 0.107*10*30

    final = (reseb+reslpg+reswat+resarea+int(transcar) +
             int(transbike)+int(trans))*12/1000
    final = '%.5f' % final
    data = {'Task': 'Hours per Day', 'electricity': reseb, 'LPG': reslpg, 'Water': reswat,
            'Area': resarea}
    data1 = {'Task': 'Hours per Day', 'Car': transcar,
             'Bike': transbike, 'Public transport': trans}
    return render_template('pie.html', data=data, data1=data1, final=final)


@app.route('/test', methods=['GET', 'POST'])
def test():
    global user
    if request.method == 'POST':
        if request.form['option1'] == '1':
            return render_template('homec.html', user=user)
        elif request.form['option1'] == '2':
            return render_template('imgind.html')

        else:
            return render_template('op2.html')
    return render_template('test.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("F:\\animal prediction\\uploads\\image.png")
        img.save("uploads\\image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        # pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()
        result = preds
        # Serialize the result, you can add additional fields
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0')
