from flask import Flask, render_template, request, jsonify  # استيراد مكتبات فلاسك لإنشاء تطبيق الويب
import joblib  # استيراد مكتبة joblib لتحميل النموذج المدرب
import numpy as np  # استيراد مكتبة numpy للتعامل مع المصفوفات والأرقام
import os  # استيراد مكتبة os للتعامل مع ملفات النظام

app = Flask(__name__)  # إنشاء كائن تطبيق فلاسك

# --- تحميل النموذج والمقياس (Model & Scaler) ---
MODEL_FILE = 'model.pkl'  # اسم ملف النموذج المدرب
SCALER_FILE = 'scaler.pkl'  # اسم ملف المقياس (لتحجيم البيانات)
ACCURACY_FILE = 'model_accuracy.txt'  # اسم الملف الذي يحتوي على دقة النموذج

# متغيرات عامة لتخزين النموذج والمقياس والدقة
model = None
scaler = None
model_accuracy = "N/A"

def load_assets():
    """
    دالة لتحميل النموذج والمقياس من الملفات.
    يتم استدعاؤها عند بدء التشغيل وعند كل طلب للصفحة الرئيسية للتأكد من تحديث النموذج.
    """
    global model, scaler, model_accuracy
    # التحقق من وجود ملفات النموذج والمقياس
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)  # تحميل النموذج
        scaler = joblib.load(SCALER_FILE)  # تحميل المقياس
    
    # التحقق من وجود ملف الدقة وقراءته
    if os.path.exists(ACCURACY_FILE):
        with open(ACCURACY_FILE, 'r') as f:
            model_accuracy = f.read().strip()
            try:
                # تحويل الدقة إلى نسبة مئوية (مثلاً 0.85 تصبح 85.00%)
                model_accuracy = f"{float(model_accuracy) * 100:.2f}%"
            except:
                pass

# تحميل الأصول عند بدء التشغيل
load_assets()

# --- الصفحة الرئيسية (Home Page) ---
@app.route('/')
def index():
    """
    دالة الصفحة الرئيسية.
    تقوم بتحميل ملف index.html وتعرضه للمستخدم.
    تمرر أيضاً دقة النموذج لعرضها في الصفحة.
    """
    # إعادة تحميل الأصول في حال تم إعادة تدريب النموذج أثناء تشغيل التطبيق
    load_assets()
    return render_template('index.html', accuracy=model_accuracy)

# --- رابط التنبؤ (Prediction API) ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    دالة التنبؤ.
    تستقبل البيانات من المستخدم بصيغة JSON،
    وتقوم بمعالجتها ثم استخدام النموذج للتنبؤ بالنتيجة.
    """
    global model, scaler
    # التأكد من أن النموذج مدرب ومحمل
    if not model or not scaler:
        return jsonify({'error': 'Model not trained yet. Please run train_model.py.'}), 500

    try:
        data = request.json  # استلام البيانات المرسلة من المتصفح
        
        # --- معالجة البيانات (Preprocessing) ---
        # استخراج العمر وتحويله إلى فئات (كما فعلنا في التدريب)
        age = float(data['age'])
        age_young = 1 if age < 30 else 0
        age_middle = 1 if 30 <= age <= 50 else 0
        age_senior = 1 if age > 50 else 0

        # تجميع كل الخصائص في قائمة واحدة بالترتيب الصحيح
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetesPedigreeFunction']),
            age_young,
            age_middle,
            age_senior
        ]
        
        # تحويل القائمة إلى مصفوفة numpy (لأن النموذج يتوقع مصفوفة ثنائية الأبعاد)
        features_array = np.array([features])
        
        # تحجيم البيانات باستخدام نفس المقياس المستخدم في التدريب
        features_scaled = scaler.transform(features_array)
        
        # --- التنبؤ (Prediction) ---
        # التنبؤ بالفئة (0 أو 1)
        prediction = model.predict(features_scaled)[0]
        # حساب احتمالية الإصابة (رقم بين 0 و 1)
        probability = model.predict_proba(features_scaled)[0][1]
        
        # تحويل النتيجة الرقمية إلى نص مفهوم
        result = "Positive (Diabetic)" if prediction == 1 else "Negative (Non-Diabetic)"
        
        # إرجاع النتيجة للمستخدم بصيغة JSON
        return jsonify({
            'prediction': result,
            'probability': f"{probability:.2f}"
        })

    except Exception as e:
        # في حال حدوث أي خطأ، يتم إرجاع رسالة الخطأ
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # تشغيل التطبيق في وضع التصحيح (Debug Mode)
    app.run(debug=True)
