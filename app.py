import os
import cv2
import csv
import zipfile
import shutil
import time
from io import BytesIO, StringIO
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, session, flash, send_file, Response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from flask_sqlalchemy import SQLAlchemy

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
app.secret_key = 'final_year_project_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pan_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Increased to 50MB to support Bulk ZIP uploads
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

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

print("Loading VGG16 model...")
model = load_model('pan_detector.keras')

vgg_base = None
vgg_idx = -1
for idx, layer in enumerate(model.layers):
    if 'vgg16' in layer.name.lower():
        vgg_base = layer
        vgg_idx = idx
        break

if vgg_base is None: raise ValueError("Could not find VGG16 base model layer.")

last_conv_layer = vgg_base.get_layer('block5_conv3')
pool_layer = vgg_base.get_layer('block5_pool')
classifier_layers = model.layers[vgg_idx + 1:]

def get_gradcam_heatmap(img_array, model, vgg_base, last_conv_layer, pool_layer):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    grad_model = tf.keras.models.Model(inputs=vgg_base.input, outputs=[last_conv_layer.output, pool_layer.output])
    
    with tf.GradientTape() as tape:
        conv_output, pool_output = grad_model(img_tensor)
        tape.watch(conv_output)
        x = pool_output
        for layer in classifier_layers:
            if isinstance(layer, tf.keras.layers.Dropout): x = layer(x, training=False)
            else: x = layer(x)
        preds = x
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0: heatmap /= max_val
    return heatmap.numpy()

def process_and_log_image(filepath, filename, username):
    """Core Inference Pipeline Refactored for Single and Batch Use"""
    img = cv2.imread(filepath)
    if img is None: return None
    
    original_h, original_w = img.shape[:2]
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype("float") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    is_tampered = True if prediction > 0.5 else False
    confidence = round(float((prediction if is_tampered else (1 - prediction)) * 100), 2)

    heatmap = get_gradcam_heatmap(img_array, model, vgg_base, last_conv_layer, pool_layer)
    heatmap_resized = cv2.resize(heatmap, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, 0.5, img, 0.5, 0)
    
    heatmap_filename = "heatmap_" + filename
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
    cv2.imwrite(heatmap_path, superimposed_img)

    record = ScanRecord(
        username=username,
        filename=filename,
        heatmap_filename=heatmap_filename,
        is_tampered=is_tampered,
        confidence=confidence
    )
    db.session.add(record)
    db.session.commit()

    return {
        'id': record.id,
        'is_tampered': is_tampered,
        'confidence': confidence,
        'original_image': filename,
        'heatmap_image': heatmap_filename
    }

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user'] = user.username
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
        else:
            db.session.add(User(username=username, password=generate_password_hash(request.form['password'])))
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
    if 'user' not in session: return redirect(url_for('login'))
    
    result_data = None
    if request.method == 'POST' and 'pan_card' in request.files:
        file = request.files.get('pan_card')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_data = process_and_log_image(filepath, filename, session['user'])

    history = ScanRecord.query.filter_by(username=session['user']).order_by(ScanRecord.timestamp.desc()).all()
    
    total_scans = len(history)
    tampered_count = sum(1 for r in history if r.is_tampered)
    
    stats = {
        'total': total_scans,
        'tampered': tampered_count,
        'authentic': total_scans - tampered_count,
        'tampered_rate': round((tampered_count / total_scans * 100), 1) if total_scans > 0 else 0.0,
        'avg_confidence': round(sum(r.confidence for r in history) / total_scans, 2) if total_scans > 0 else 0.0
    }

    return render_template('index.html', result=result_data, history=history, stats=stats, user=session['user'])

@app.route('/batch_process', methods=['POST'])
def batch_process():
    """Extracts a ZIP file, processes all images, and returns a combined CSV report."""
    if 'user' not in session: return redirect(url_for('login'))
    
    file = request.files.get('zip_file')
    if not file or not file.filename.lower().endswith('.zip'):
        flash('Please upload a valid .zip file', 'danger')
        return redirect(url_for('index'))
        
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(zip_path)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(temp_dir)
        flash('Invalid or corrupted ZIP file', 'danger')
        return redirect(url_for('index'))
        
    results = []
    valid_exts = ('.jpg', '.jpeg', '.png')
    
    for root, _, files in os.walk(temp_dir):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                original_path = os.path.join(root, fname)
                safe_name = f"batch_{int(time.time())}_{secure_filename(fname)}"
                final_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
                
                shutil.copy(original_path, final_path)
                res = process_and_log_image(final_path, safe_name, session['user'])
                if res: results.append(res)
                
    shutil.rmtree(temp_dir) # Clean up temp extracted files
    flash(f'Batch complete! Successfully processed {len(results)} documents.', 'success')

    # Generate CSV Output directly
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Record ID', 'Compliance Agent', 'Timestamp (UTC)', 'Original Filename', 'Status', 'Confidence Score (%)'])
    for r in results:
        writer.writerow([f"#{r['id']:05d}", session['user'], datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), r['original_image'], 'Tampered' if r['is_tampered'] else 'Authentic', f"{r['confidence']}%"])
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": f"attachment; filename=bulk_batch_report_{int(time.time())}.csv"}
    )

@app.route('/export_csv')
def export_csv():
    if 'user' not in session: return redirect(url_for('login'))
    history = ScanRecord.query.filter_by(username=session['user']).order_by(ScanRecord.timestamp.desc()).all()
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Record ID', 'Compliance Agent', 'Timestamp (UTC)', 'Original Filename', 'Status', 'Confidence Score (%)', 'Grad-CAM Heatmap File'])
    
    for record in history:
        writer.writerow([f"#{record.id:05d}", record.username, record.timestamp.strftime('%Y-%m-%d %H:%M:%S'), record.filename, 'Tampered' if record.is_tampered else 'Authentic', f"{record.confidence:.2f}%", record.heatmap_filename])
        
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv', headers={"Content-Disposition": f"attachment; filename=kyc_audit_log_{session['user']}.csv"})

@app.route('/download_report/<int:record_id>')
def download_report(record_id):
    if 'user' not in session: return redirect(url_for('login'))
    record = ScanRecord.query.get_or_404(record_id)
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#0F172A'), alignment=1)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#64748B'), alignment=1)
    
    elements = []
    elements.append(Paragraph("KYC FORENSIC AUDIT REPORT", title_style))
    elements.append(Paragraph("Automated PAN Card Tampering & Forgery Detection Portal", subtitle_style))
    elements.append(Spacer(1, 15))
    
    status_text = "TAMPERED / FORGERY DETECTED" if record.is_tampered else "AUTHENTIC DOCUMENT"
    status_color = "#E11D48" if record.is_tampered else "#059669"
    
    meta_data = [
        [Paragraph("<b>Audit ID:</b>", styles['Normal']), Paragraph(f"#{record.id:05d}", styles['Normal'])],
        [Paragraph("<b>Agent:</b>", styles['Normal']), Paragraph(record.username, styles['Normal'])],
        [Paragraph("<b>Timestamp:</b>", styles['Normal']), Paragraph(record.timestamp.strftime('%Y-%m-%d %H:%M:%S'), styles['Normal'])],
        [Paragraph("<b>Filename:</b>", styles['Normal']), Paragraph(record.filename, styles['Normal'])],
        [Paragraph("<b>Verdict:</b>", styles['Normal']), Paragraph(f"<font color='{status_color}'><b>{status_text}</b></font>", styles['Normal'])],
        [Paragraph("<b>Confidence:</b>", styles['Normal']), Paragraph(f"<b>{record.confidence:.2f}%</b>", styles['Normal'])],
    ]
    
    t1 = Table(meta_data, colWidths=[2.2*inch, 4.8*inch])
    t1.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')), ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')), ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')), ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5), ('LEFTPADDING', (0,0), (-1,-1), 10)]))
    elements.append(t1)
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("<b>Visual Forensic Examination:</b>", styles['Heading3']))
    elements.append(Spacer(1, 8))
    
    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], record.filename)
    heat_path = os.path.join(app.config['UPLOAD_FOLDER'], record.heatmap_filename)
    
    orig_img = RLImage(orig_path, width=3.3*inch, height=2.2*inch) if os.path.exists(orig_path) else Paragraph("Original missing", styles['Normal'])
    heat_img = RLImage(heat_path, width=3.3*inch, height=2.2*inch) if os.path.exists(heat_path) else Paragraph("Heatmap missing", styles['Normal'])
    
    t2 = Table([[Paragraph("<b>Input Document</b>", styles['Normal']), Paragraph("<b>Grad-CAM Heatmap</b>", styles['Normal'])], [orig_img, heat_img]], colWidths=[3.5*inch, 3.5*inch])
    t2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    elements.append(t2)
    
    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"audit_report_{record.id}.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)