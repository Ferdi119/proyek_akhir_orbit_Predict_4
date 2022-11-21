'''
	Contoh Deloyment untuk Domain Computer Vision (CV)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask, render_template, request, jsonify, redirect, Response
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    Flatten, Dense, Activation, Dropout, LeakyReLU
from PIL import Image
from fungsi import make_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as ts

import argparse
import io
from PIL import Image
import cv2
import torch

model_yolo = torch.hub.load("ultralytics/yolov5", "custom", path = "./best.pt", force_reload=True)
from io import BytesIO

def gen():
    cap=cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = model_yolo(img, size=640)
            #print(results)
            #print(results.pandas().xyxy[0])
            #results.render()  # updates results.imgs with boxes and labels
            results.print()  # print results to screen
            #results.show() 
            #print(results.imgs)
            #print(type(img))
            #print(results)
            #plt.imshow(np.squeeze(results.render()))
            #print(type(img))
            #print(img.mode)
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

            #print(type(img))
            #print(img.shape)
            #frame = img
            #ret,buffer=cv2.imencode('.jpg',img)
            #frame=buffer.tobytes()
            #print(type(frame))
            #for img in results.imgs:
                #img = Image.fromarray(img)
            #ret,img=cv2.imencode('.jpg',img)
            #img=img.tobytes()

            #encode output image to bytes
            #img = cv2.imencode('.jpg', img)[1].tobytes()
            #print(type(img))
        else:
            break
        #print(cv2.imencode('.jpg', img)[1])

        #print(b)
        #frame = img_byte_arr

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def PredGambar(file_gmbr):
    file = file_gmbr
    gmbr_array = np.asarray(file)
    gmbr_array = gmbr_array*(1/225)
    gmbr_input = tf.reshape(gmbr_array, shape=[1, 150, 150, 3])

    predik_array = model.predict(gmbr_input)[0]

    df = pd.DataFrame(predik_array)
    df = df.rename({0: 'NilaiKemiripan'}, axis='columns')
    Kualitas = ['AyamSegar', 'AyamTiren']
    df['Kelas'] = Kualitas
    df = df[['Kelas', 'NilaiKemiripan']]

    predik_kelas = np.argmax(model.predict(gmbr_input))

    if predik_kelas == 0:
        predik_Kualitas = 'AyamSegar'
    else:
        predik_Kualitas = 'AyamTiren'

    return predik_Kualitas, df

# =[Variabel Global]=============================


app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 22500 * 22500
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.JPG']
app.config['UPLOAD_PATH'] = './static/images/uploads/'

# model = None

NUM_CLASSES = 2
cifar10_classes = ["AyamSegar", "AyamTiren"]

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]


@app.route("/")
def beranda():
    return render_template('index.html')

@app.route("/beranda")
def beranda_2():
    return render_template('index.html')

# [Routing untuk API]


@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi
    hasil_prediksi = '(none)'
    gambar_prediksi = '(none)'

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)

    # Periksa apakah ada file yg dipilih untuk diupload
    if filename != '':

        # Set/mendapatkan extension dan path dari file yg diupload
        file_ext = os.path.splitext(filename)[1]
        gambar_prediksi = '/static/images/uploads/' + filename

        # Periksa apakah extension file yg diupload sesuai (jpg)
        if file_ext in app.config['UPLOAD_EXTENSIONS']:

            # Simpan Gambar
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_PATH'], filename))

            # Memuat Gambar
            lok = '.' + gambar_prediksi
            gmbr = ts.keras.utils.load_img(lok, target_size=(150, 150))
            x = ts.keras.utils.img_to_array(gmbr)
            x = np.expand_dims(x, axis=0)
            gmbr = np.vstack([x])

            # Prediksi Gambar
            kelas, df = PredGambar(gmbr)
            hasil_prediksi = kelas

            # Return hasil prediksi dengan format JSON
            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })
        else:
            # Return hasil prediksi dengan format JSON
            gambar_prediksi = '(none)'
            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })
@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/objectDetection')
def objectDetection():
    return render_template('detection.html')
            

# =[Main]========================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # Load model yang telah ditraining
    model = make_model()
    model.load_weights("AyamDenseNet201-DanielMrnth_2.h5")
    # model.load_weights("model_cifar10_cnn_tf.h5")

    # Run Flask di localhost
    app.run(host="localhost", port=5000, debug=True)
