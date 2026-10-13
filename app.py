import os
import cv2
import csv
from io import BytesIO, StringIO
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, session, flash, send_file, Response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from flask_sqlalchemy import SQLAlchemy

# ReportLab components for PDF generation
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
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

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
        file = request.files.get('pan_card')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            original_h, original_w = img.shape[:2]
            
            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            img_array = img_rgb.astype("float") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
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
                username=session['user'],
                filename=filename,
                heatmap_filename=heatmap_filename,
                is_tampered=is_tampered,
                confidence=confidence
            )
            db.session.add(record)
            db.session.commit()

            result_data = {
                'id': record.id,
                'is_tampered': is_tampered,
                'confidence': confidence,
                'original_image': filename,
                'heatmap_image': heatmap_filename
            }

    history = ScanRecord.query.filter_by(username=session['user']).order_by(ScanRecord.timestamp.desc()).all()
    
    total_scans = len(history)
    tampered_count = sum(1 for r in history if r.is_tampered)
    authentic_count = total_scans - tampered_count
    tampered_rate = round((tampered_count / total_scans * 100), 1) if total_scans > 0 else 0.0
    avg_confidence = round(sum(r.confidence for r in history) / total_scans, 2) if total_scans > 0 else 0.0

    stats = {
        'total': total_scans,
        'tampered': tampered_count,
        'authentic': authentic_count,
        'tampered_rate': tampered_rate,
        'avg_confidence': avg_confidence
    }

    return render_template('index.html', result=result_data, history=history, stats=stats, user=session['user'])

# Feature 1: Export Audit Trail to CSV
@app.route('/export_csv')
def export_csv():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    history = ScanRecord.query.filter_by(username=session['user']).order_by(ScanRecord.timestamp.desc()).all()
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Record ID', 'Compliance Agent', 'Timestamp (UTC)', 'Original Filename', 'Status', 'Confidence Score (%)', 'Grad-CAM Heatmap File'])
    
    for record in history:
        writer.writerow([
            f"#{record.id:05d}",
            record.username,
            record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            record.filename,
            'Tampered' if record.is_tampered else 'Authentic',
            f"{record.confidence:.2f}%",
            record.heatmap_filename
        ])
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": f"attachment; filename=kyc_audit_log_{session['user']}.csv"}
    )

# Feature 2: Download Forensic PDF Audit Report
@app.route('/download_report/<int:record_id>')
def download_report(record_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    record = ScanRecord.query.get_or_404(record_id)
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748B'),
        alignment=1
    )
    normal_style = styles['Normal']
    
    elements = []
    
    elements.append(Paragraph("KYC FORENSIC AUDIT REPORT", title_style))
    elements.append(Paragraph("Automated PAN Card Tampering & Forgery Detection Portal", subtitle_style))
    elements.append(Spacer(1, 15))
    
    status_text = "TAMPERED / FORGERY DETECTED" if record.is_tampered else "AUTHENTIC DOCUMENT"
    status_color = "#E11D48" if record.is_tampered else "#059669"
    
    meta_data = [
        [Paragraph("<b>Audit Record ID:</b>", normal_style), Paragraph(f"#{record.id:05d}", normal_style)],
        [Paragraph("<b>Verifying Agent:</b>", normal_style), Paragraph(record.username, normal_style)],
        [Paragraph("<b>Timestamp (UTC):</b>", normal_style), Paragraph(record.timestamp.strftime('%Y-%m-%d %H:%M:%S'), normal_style)],
        [Paragraph("<b>Document Filename:</b>", normal_style), Paragraph(record.filename, normal_style)],
        [Paragraph("<b>Forensic Verdict:</b>", normal_style), Paragraph(f"<font color='{status_color}'><b>{status_text}</b></font>", normal_style)],
        [Paragraph("<b>Model Confidence:</b>", normal_style), Paragraph(f"<b>{record.confidence:.2f}%</b> (VGG16 Transfer Learning)", normal_style)],
    ]
    
    meta_table = Table(meta_data, colWidths=[2.2 * inch, 4.8 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("<b>Visual Forensic Examination (Side-by-Side):</b>", styles['Heading3']))
    elements.append(Spacer(1, 8))
    
    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], record.filename)
    heat_path = os.path.join(app.config['UPLOAD_FOLDER'], record.heatmap_filename)
    
    orig_img = RLImage(orig_path, width=3.3 * inch, height=2.2 * inch) if os.path.exists(orig_path) else Paragraph("Original file missing", normal_style)
    heat_img = RLImage(heat_path, width=3.3 * inch, height=2.2 * inch) if os.path.exists(heat_path) else Paragraph("Heatmap file missing", normal_style)
    
    img_data = [
        [Paragraph("<b>Input Document (RGB)</b>", normal_style), Paragraph("<b>Grad-CAM Activation Map (block5_conv3)</b>", normal_style)],
        [orig_img, heat_img]
    ]
    
    img_table = Table(img_data, colWidths=[3.5 * inch, 3.5 * inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 25))
    
    disclaimer = Paragraph(
        "<font size='8' color='#94A3B8'>CONFIDENTIAL COMPLIANCE DOCUMENT: Generated by Automated Document Forgery Detection Engine. Model activations traced via Gradient-weighted Class Activation Mapping (Grad-CAM). For certified compliance purposes only.</font>",
        subtitle_style
    )
    elements.append(disclaimer)
    
    doc.build(elements)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"audit_report_{record.id}_{record.filename}.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)