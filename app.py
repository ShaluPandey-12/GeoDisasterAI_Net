from flask import Flask, request, render_template, url_for , redirect
import numpy as np
import cv2
import os
import uuid
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from lime import lime_image
from skimage.segmentation import mark_boundaries
import re
import sqlite3

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "static/results/uploads"
GRADCAM_FOLDER = "static/results/gradcam"
HEATMAP_FOLDER = "static/results/heatmaps"
LIME_FOLDER = "static/results/lime"

for folder in [UPLOAD_FOLDER, GRADCAM_FOLDER, HEATMAP_FOLDER, LIME_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load model
model = load_model("Models/ensemble_hybrid.h5", compile=False)
submodel = model.get_layer("model")

class_names = ["CYCLONE", "EARTHQUAKE", "FLOOD", "WILDFIRE"]

# Locate last conv layer
last_conv = None
for layer in reversed(submodel.layers):
    if len(layer.output_shape) == 4:
        last_conv = layer.name
        break
print("Last conv layer:", last_conv)

# -------- GRAD-CAM FUNCTION --------
def make_gradcam(model, img_tensor, class_idx):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    return heatmap

# -------- ROUTES --------


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    # Save uploaded image
    file_id = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_FOLDER, file_id + ".jpg")
    file.save(img_path)

    # Preprocess
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_preprocessed = preprocess_input(img_array)
    input_tensor = np.expand_dims(img_preprocessed, axis=0)

    # Prediction
    pred = model.predict(input_tensor)[0]
    predicted_class = np.argmax(pred)
    predicted_label = class_names[predicted_class]
    # Confidence for the predicted class
    confidence = float(pred[predicted_class])

    # ----------------------------- GRAD-CAM -----------------------------
    heatmap = make_gradcam(submodel, input_tensor, predicted_class)

    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, (128, 128))

    # Heatmap visualization
    heatmap_resized = cv2.resize(heatmap, (128, 128))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_path = os.path.join(HEATMAP_FOLDER, file_id + "_heatmap.jpg")
    cv2.imwrite(heatmap_path, heatmap_color)

    # Heatmap overlay
    gradcam_overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
    gradcam_path = os.path.join(GRADCAM_FOLDER, file_id + "_gradcam.jpg")
    cv2.imwrite(gradcam_path, gradcam_overlay)

    # ------------------------------- LIME --------------------------------
    def predict_fn(images):
        images = preprocess_input(np.array(images).astype(np.float32))
        return model.predict(images)

    orig_img = img_array.astype("double")
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        orig_img,
        predict_fn,
        top_labels=4,
        hide_color=0,
        num_samples=500
    )

    lime_img, mask = explanation.get_image_and_mask(
        label=predicted_class,
        positive_only=True,
        hide_rest=False,
        num_features=5,
        min_weight=0.0
    )

    boundary_img = mark_boundaries(orig_img.astype("uint8"), mask)
    lime_bgr = cv2.cvtColor((boundary_img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)

    lime_path = os.path.join(LIME_FOLDER, file_id + "_lime.jpg")
    cv2.imwrite(lime_path, lime_bgr)
 

    # ------------------------ RETURN RESULT ------------------------
    return render_template(
        "result.html",
        prediction=predicted_label,
        confidence=confidence,
        original=url_for("static", filename="results/uploads/" + file_id + ".jpg"),
        heatmap=url_for("static", filename="results/heatmaps/" + file_id + "_heatmap.jpg"),
        gradcam=url_for("static", filename="results/gradcam/" + file_id + "_gradcam.jpg"),
        lime=url_for("static", filename="results/lime/" + file_id + "_lime.jpg"),
    )
    print(gradcam)



@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('signin.html')

@app.route('/logon')
def logon():
    return render_template('signup.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route("/graphs")
def about1():
    return render_template("graphs.html")

@app.route("/Notebook")
def about2():
    return render_template("Notebook.html")

@app.route("/Notebook2")
def about3():
    return render_template("Notebook2.html")

@app.route("/Notebook3")
def about4():
    return render_template("Notebook3.html")



if __name__ == "__main__":
    app.run(debug=False)
