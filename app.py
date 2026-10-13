import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'final_year_project_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pan_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class ScanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    filename = db.Column(db.String(120), nullable=False)
    heatmap_filename = db.Column(db.String(120), nullable=False)
    is_tampered = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# Load the trained VGG16 model safely
print("Loading VGG16 model...")
model = load_model('pan_detector.keras')

vgg_base = None
vgg_idx = -1
for idx, layer in enumerate(model.layers):
    if 'vgg16' in layer.name.lower():
        vgg_base = layer
        vgg_idx = idx
        break

if vgg_base is None:
    raise ValueError("Could not find VGG16 base model layer.")

last_conv_layer = vgg_base.get_layer('block5_conv3')
pool_layer = vgg_base.get_layer('block5_pool')
classifier_layers = model.layers[vgg_idx + 1:]

def get_gradcam_heatmap(img_array, model, vgg_base, last_conv_layer, pool_layer):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    grad_model = tf.keras.models.Model(
        inputs=vgg_base.input,
        outputs=[last_conv_layer.output, pool_layer.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, pool_output = grad_model(img_tensor)
        tape.watch(conv_output)
        
        x = pool_output
        for layer in classifier_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=False)
            else:
                x = layer(x)
        preds = x
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_output)
    
    if grads is None:
        raise ValueError("Gradient calculation failed for VGG16 model.")
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
        
    return heatmap.numpy()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user'] = username
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    result_data = None
    if request.method == 'POST':
        file = request.files['pan_card']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            original_h, original_w = img.shape[:2]
            
            img_resized = cv2.resize(img, (224, 224))
            
            # Convert BGR to RGB to match VGG16 ImageNet training standards
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            img_array = img_rgb.astype("float") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            is_tampered = True if prediction > 0.5 else False
            confidence = round((prediction if is_tampered else (1 - prediction)) * 100, 2)

            heatmap = get_gradcam_heatmap(img_array, model, vgg_base, last_conv_layer, pool_layer)
            
            # Bicubic interpolation for smooth, high-resolution visual output
            heatmap_resized = cv2.resize(heatmap, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
            heatmap_resized = np.uint8(255 * heatmap_resized)
            
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(heatmap_color, 0.5, img, 0.5, 0)
            
            heatmap_filename = "heatmap_" + filename
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
            cv2.imwrite(heatmap_path, superimposed_img)

            record = ScanRecord(
                username=session['user'],
                filename=filename,
                heatmap_filename=heatmap_filename,
                is_tampered=is_tampered,
                confidence=confidence
            )
            db.session.add(record)
            db.session.commit()

            result_data = {
                'is_tampered': is_tampered,
                'confidence': confidence,
                'original_image': filename,
                'heatmap_image': heatmap_filename
            }

    history = ScanRecord.query.filter_by(username=session['user']).order_by(ScanRecord.timestamp.desc()).all()
    return render_template('index.html', result=result_data, history=history, user=session['user'])

if __name__ == '__main__':
    app.run(debug=True)