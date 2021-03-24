import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
global graph
graph = tf.compat.v1.get_default_graph()
from flask import Flask , request, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

    
@app.route('/predict',methods = ['GET','POST'])
def preds():
    CATEGORIES = ["You are Normal! :)","You have Pneumonia!! :("]
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        img=np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img=img/255.0
        
        with graph.as_default():
            model = load_model(r"F:\Project\spyder\models\cnnmodel.h5")
            p=model.predict(img)
            p=model.predict_classes(img)
            print(p)

    return (CATEGORIES[int(p[0][0])])

if  __name__ == "__main__":
    app.run(debug=True)