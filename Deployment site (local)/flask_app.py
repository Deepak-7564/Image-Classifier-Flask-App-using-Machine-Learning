from flask import Flask, render_template
from flask import request
import os
import pickle
import scipy
import numpy as np
import pandas as ps
import sklearn
import skimage
import skimage.io
import skimage.feature
import skimage.transform
import skimage.color
from sklearn.pipeline import make_pipeline


app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/')

# ----------Load Model-----------------------
model_sgd_path = os.path.join(MODEL_PATH,'dsa_image_classification_sgd.pickle')
scaler_path = os.path.join(MODEL_PATH,'dsa_scaler.pickle')
model_sgd = pickle.load(open(model_sgd_path,'rb'))
model = model_sgd
scaler = pickle.load(open(scaler_path,'rb'))

@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 : Page Not Found !!! ,Please go to the home page and try again "
    return render_template("error.html",message=message)

@app.errorhandler(405)
def error405(error):
    message = "ERROR 405 : Method Not Found !!!"
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message = 'INTERNAL ERROR 500 : Error occurs in the program !!!'
    return render_template("error.html",message=message)



@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename
        print("The filename that has been uploaded = ",filename)
        # Know the extension of filename
        # All only .jpg, .png, .jpeg
        ext = filename.split('.')[-1]
        print('The extension of the filename = ',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            #saveing the image
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File save successfully')
            #send to pipeeline model
            results = pipeline_model(path_save,scaler,model_sgd)
            hei = getheight(path_save)
            print(results)
            return render_template('upload.html',fileupload=True,extension=False,data=results,image_filename=filename,height=hei)

        else:
            print('use only the extension with .jpg, .png, .jpeg')
            return render_template('upload.html',extension=True,fileupload=False)
        
    else:
        return render_template('upload.html',fileupload=False,extension=False)
    
@app.route('/about/')
def about():
    return render_template('about.html')

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ = img.shape
    aspect = h/w
    given_weidth = 300
    height =given_weidth*aspect
    return height

def pipeline_model(path,scaler_transform,model_sgd):
    #pipeline
    image = skimage.io.imread(path)
    #transform image into 80x80
    image_resize = skimage.transform.resize(image,(80,80))
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8)
    #rgb to gray
    gray = skimage.color.rgb2gray(image_transform)
    #hog feature
    feature_vector = skimage.feature.hog(gray,
                                     orientations=10,
                                     pixels_per_cell=(8,8),cells_per_block=(2,2))

    #Scaling
    scalex = scaler.transform(feature_vector.reshape(1,-1))
    result = model.predict(scalex)
    
    #decision function # confidence
    decision_value = model_sgd.decision_function(scalex).flatten()
    labels = model_sgd.classes_
    
    #probablity
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)
    prob_value
    
    #top 5
    top_5_prob_ind = prob_value.argsort()[::-1][:5]
    top_labels = labels[top_5_prob_ind]
    top_prob = prob_value[top_5_prob_ind]
    #put in dictornary
    top_dict = dict()
    for key,val in zip(top_labels,top_prob):
        top_dict.update({key:np.round(val,2)})
        

    return top_dict


if __name__ == '__main__':
      app.run(debug=True)