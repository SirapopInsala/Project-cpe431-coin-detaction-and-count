from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from detection import detect_objects  # ใช้ detect_objects จาก count_object.py

app = Flask(__name__, static_folder="static")  # กำหนด static folder ให้ถูกต้อง

UPLOAD_FOLDER = "object_detection_flask/static/uploads"
RESULT_FOLDER = "object_detection_flask/static/results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ตรวจสอบและสร้างโฟลเดอร์ถ้ายังไม่มี
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ตรวจสอบว่าเป็นไฟล์ภาพที่อนุญาตหรือไม่
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# หน้าเว็บหลัก
@app.route("/")
def index():
    return render_template("index.html")

COIN_VALUES = {
    1: 10,   # เหรียญ 10 บาท
    2: 5,    # เหรียญ 5 บาท
    3: 2,    # เหรียญ 2 บาท
    4: 1     # เหรียญ 1 บาท
}

# อัปโหลดและประมวลผลภาพ
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(request.url)

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Detect objects and save the result
    result_path, coin_counts, total_value = detect_objects(filepath, filename, app.config["RESULT_FOLDER"])

    # Ensure path uses forward slashes for URLs
    result_image = url_for('static', filename=f'results/{filename}')

    # Send results to result.html
    return render_template("result.html", result_image=result_image, coin_counts=coin_counts, total_value=total_value, coin_values=COIN_VALUES)

if __name__ == "__main__":
    app.run(debug=True)
