import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

model = tf.keras.models.load_model('pest_disease_model.h5')
class_names = ['โรคใบจุด', 'โรคใบไหม้', 'โรคราสนิมใบ']

if len(sys.argv) < 2:
    print("❌ กรุณาระบุ path รูปภาพ เช่น python predict_disease.py path_to_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]
print(f"DEBUG: รับ path มา = {image_path}")  # ดูพาธก่อน

if not os.path.isfile(image_path):
    print(f"❌ ไม่พบไฟล์รูปภาพ: {image_path}")
    sys.exit(1)

img = Image.open(image_path).resize((128, 128))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"\n✅ ทำนายว่าเป็น: {predicted_class}")
