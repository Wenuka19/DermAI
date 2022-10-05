
import tensorflow as tf
import numpy as np
import cv2

img_classes = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases',
               'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']


def predict():
    new_model = tf.keras.models.load_model('dermAI1.h5')
    img = cv2.imread('img04.jpg')
    resized_img = tf.image.resize(img, (256, 256))
    prediction = new_model.predict(np.expand_dims(resized_img/255, axis=0))
    top5 = np.argsort(prediction[0])[::-1][:5]
    result = ""
    for i in top5:
        result += img_classes[i]+"   " + \
            str(round(prediction[0][i]*100, 2))+"%"+"\n"
    return result
