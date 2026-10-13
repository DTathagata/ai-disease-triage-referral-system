import joblib
import numpy as np
from django.shortcuts import render
import os
from django.conf import settings

model = joblib.load(os.path.join(settings.BASE_DIR, 'ml65cd.joblib'))

SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing',
    'shivering','chills','joint_pain','stomach_pain','acidity',
    'ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition',
    'spotting_ urination','fatigue','weight_gain','anxiety',
    'cold_hands_and_feets','mood_swings','weight_loss','restlessness',
    'lethargy','patches_in_throat','irregular_sugar_level','cough',
    'high_fever','sunken_eyes','breathlessness','sweating','dehydration',
    'indigestion','headache','yellowish_skin','dark_urine','nausea',
    'loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise',
    'blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising',
    'obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
    'enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness',
    'stiff_neck','swelling_joints','movement_stiffness','spinning_movements',
    'loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching',
    'toxic_look_(typhos)','depression','irritability','muscle_pain',
    'altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes',
    'increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma',
    'stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload.1','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking',
    'pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails',
    'blister','red_sore_around_nose','yellow_crust_ooze',
]

# Maps disease name to a Google Images search URL for that disease
DISEASE_IMAGES = {
    'Acne':                                  'images/Acne.webp',
    'AIDS':                                  'images/AIDS.png',
    'Alcoholic hepatitis':                   'images/Alcoholic_hepatitis.png',
    'Allergy':                               'images/Allergy.jpg',
    'Arthritis':                             'images/Arthritis.webp',
    '(vertigo) Paroymsal  Positional Vertigo': 'images/benign-paroxysmal-positional-vertigo.jpg',
    'Bronchial Asthma':                      'images/Bronchial_Asthma.jpg',
    'Cervical spondylosis':                  'images/Cervical_Spondylosis.jpg',
    'Chicken pox':                           'images/Chicken_pox.jpg',
    'Chronic cholestasis':                   'images/Chronic_cholestasis.webp',
    'Common Cold':                           'images/Common_Cold.jpg',
    'Dengue':                                'images/Dengue.png',
    'Diabetes':                             'images/Diabetes.jpg',
    'Dimorphic hemmorhoids(piles)':          'images/Dimorphic_hemmorhoids_piles_.png',
    'Drug Reaction':                         'images/Drug_Reaction.jpg',
    'Fungal infection':                      'images/Fungal infection.jpg',
    'Gastroenteritis':                       'images/Gastroenteritis.jpg',
    'GERD':                                  'images/GERD.png',
    'Heart attack':                          'images/Heart_attack.png',
    'hepatitis A':                           'images/hepatitis_A.jpg',
    'Hepatitis B':                             'images/hepatitis B.jpg',
    'Hepatitis C':                             'images/hepatitis C.jpg',
    'Hepatitis D':                             'images/hepatitis D.webp',
    'Hepatitis E':                             'images/hepatitis E.png',
    'Hypertension':                           'images/Hypertension.jpg',
    'Hyperthyroidism':                         'images/Hyperthyroidism.jpg',
    'Hypoglycemia':                            'images/Hypoglycemia.avif',
    'Hypothyroidism':                          'images/Hypothyroidism.jpg',
    'Impetigo':                                'images/Impetigo.jpg',
    'Jaundice':                                'images/Jaundice.png',
    'Malaria':                                 'images/Malaria.webp',
    'Migraine':                                'images/Migraine.jpg',
    'Osteoarthristis':                         'images/Osteoarthristis.jpg',
    'Paralysis (brain hemorrhage)':            'images/Paralysis (brain hemorrhage).jpg',
    'Peptic ulcer diseae':                     'images/Peptic ulcer diseae.jpg',
    'Pneumonia':                               'images/Pneumonia.jpg',
    'Psoriasis':                               'images/Psoriasis.png',
    'Tuberculosis':                            'images/Tuberculosis.webp',
    'Typhoid':                                 'images/Typhoid.jpg',
    'Urinary tract infection':                 'images/Urinary tract infection.jpg',
    'Varicose veins':                          'images/Varicose veins.jpg',
}

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def predict(request):
    result = None
    image_file = None
    selected_symptoms = []

    if request.method == 'POST':
        s1 = request.POST.get('symptom1', '')
        s2 = request.POST.get('symptom2', '')
        s3 = request.POST.get('symptom3', '')
        s4 = request.POST.get('symptom4', '')
        s5 = request.POST.get('symptom5', '')
        selected_symptoms = [s for s in [s1,s2,s3,s4,s5] if s]

        features = np.array(
            [1 if s in selected_symptoms else 0 for s in SYMPTOMS]
        ).reshape(1, -1)

        prediction = model.predict(features)[0].strip()
        print(f"DEBUG prediction: '{prediction}'")    # ← ADD THIS
        result = prediction
        image_file = DISEASE_IMAGES.get(prediction, None)
        print(f"DEBUG image_file: '{image_file}'")    # ← ADD THIS

    return render(request, 'predict.html', {
        'result':     result,
        'image_file': image_file,
        'symptoms':   SYMPTOMS,
    })