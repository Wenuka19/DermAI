from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

img_classes = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases',
               'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']


def predict():
    interpreter = Interpreter(model_path='test_model.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = Image.open('/home/pi/Pictures/output.jpg')
    resized_img = np.array(img.resize((256, 256)),dtype=np.float32)
    input_data = np.expand_dims(resized_img/255,axis=0)
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    top5 = np.argsort(prediction)[::-1][:5]
    result = []
    for i in top5:
        result.append([img_classes[i],str(round(prediction[i]*100, 2))+"%"])
#         result += img_classes[i]+"   " + str(round(prediction[i]*100, 2))+"%"+"\n"
    return result
print(predict())