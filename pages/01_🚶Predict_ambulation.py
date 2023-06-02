import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model


@st.cache(allow_output_mutation=True)
def load_model_for_cache():
    ret = load_model('finalized_classification_model')
    return ret    


st.title('Prediction for functional ambulation')
st.write('This is a web app for predicting functional ambulation in patients with spinal cord injury at the time of discharge from a rehabilitation hospital, as evaluated by the locomotion (walk/wheelchair) portion of the Functional Independence Measure (FIM) scale. Functional ambulation is defined as a score of 6 (modified independence) or 7 (complete independence) in which the mode of ambulation was either walk or using walking and a wheelchair equally.')

st.write('The default value of each feature in the web application is a placeholder. Users must modify their inputs according to the clinical characteristics of each individual patient. Users should be aware that the algorithm requires complete information about the items contained in the interface to run the algorithm. Once the input is completed, users can click the “Predict” button at the bottom to see the prediction. The app will tell users, with a probability score, whether patient ambulation will be independent or dependent.')

st.write('PLEASE NOTE: Hitting the + button or - button repeatedly, the app may crash.')
#dataset.csvの特徴量の変更する


#Age = st.sidebar.slider(label = 'Age (18-9)', min_value = 18,
                          #max_value = 99 ,
                          #value = 52,
                          #step = 1)

Age = st.sidebar.number_input(label = 'Age　(18-99)',
                          min_value = 18,
                          max_value = 99,
                          value = 52,
                          step = 1
                          )

#Days_from_injury_to_admission = st.sidebar.slider(label = 'Days from injury to admission', min_value = 1,
                          #max_value = 365 ,
                          #value = 74,
                          #step = 1)

Days_from_injury_to_admission = st.sidebar.number_input(label = 'Days from injury to admission (1-428) ',
                          min_value = 1,
                          max_value = 428,
                          value = 30,
                          step = 1
                          )


FIM_walk_wheelchair = st.sidebar.number_input(label = 'FIM walk/wheelchair (1-7)',
                          min_value = 1,
                          max_value = 7,
                          value = 2,
                          step = 1 )

FIM_problem_solving = st.sidebar.number_input(label = 'FIM problem solving (1-7)',
                          min_value = 1,
                          max_value = 7,
                          value = 7,
                          step = 1)
#FIM_social_interaction = st.sidebar.number_input(label = 'FIM social interaction (1-7)',
  #                        min_value = 1,
#                          max_value = 7,
 #                         value = 7,
  #                        step = 1)


Total_FIM_score = st.sidebar.number_input(label = 'Total FIM score (18-126)', min_value = 18,
                          max_value = 126,
                          value = 60,
                          step = 1)

ASIA_motor_score = st.sidebar.number_input(label = 'ASIA motor score (0-100)', min_value = 0,
                          max_value = 100 ,
                          value = 60,
                          step = 1)

#Sensory_score_light_touch = st.sidebar.number_input(label = 'Sensory score light touch (0-112)', min_value = 0,#
#                          max_value = 112 ,
#                          value = 60,
#                          step = 1)

Educational_background = st.sidebar.selectbox('Educational background (final education)',
                          ("Elementary school", "Junior high school", "High school", "University", "Graduate school"))

Educational_background_dic = {"Elementary school": 2, "Junior high school": 3, "High school": 4, "University": 5, "Graduate school": 6}

Occupation_input = st.sidebar.selectbox('Occupation',
                          ("Housemaker", "Student", "Unemployed", "Retired", "Employed"))

Occupation_dic = {"Housemaker": 0, "Student": 0, "Unemployed": 1, "Retired": 1, "Employed": 0}

# Neurological_level_of_injury = st.sidebar.selectbox('Neurological level of injury',
#                    ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "T1", "T2", "T3",
#                     "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1",
#                     "L2", "L3", "L4", "L5", "S1", "S2", "S3", "S4"))

# NLI_value_dic = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8,
#                  "T1": 9, "T2": 10, "T3": 11, "T4": 12, "T5": 13, "T6": 14, "T7": 15,
#                  "T8": 16, "T9": 17, "T10": 18, "T11": 19, "T12": 20, "L1": 21, "L2": 22,
#                  "L3": 23, "L4": 24, "L5": 25, "S1": 26, "S2": 27, "S3": 28, "S4": 29}


#st.table(features_df)

if st.button('Predict'):
    model = load_model_for_cache()

    features = {'Age': Age,
            'Days_from_injury_to_admission': Days_from_injury_to_admission,
            'FIM_walk_wheelchair': FIM_walk_wheelchair,
            #'FIM_social_interaction': FIM_social_interaction,
            'FIM_problem_solving': FIM_problem_solving,
            'Total_FIM_score': Total_FIM_score,
            'Educational_background': Educational_background_dic[Educational_background],
            #'Neurological_level_of_injury': NLI_value_dic[Neurological_level_of_injury],
            'ASIA_motor_score': ASIA_motor_score,
            'Occupation_12.0': Occupation_dic[Occupation_input]
           # 'Sensory_score_light_touch': Sensory_score_light_touch
            }


    features_df  = pd.DataFrame([features])

    no_use = {
        'Sex': 1,
        'FIM eating': 1,
        'FIM grooming': 1,
        'FIM bathing': 1,
        'FIM dressing lower body': 1,
        'FIM dressing upper body': 1,
        'FIM toileting': 1,
        'FIM bladder management': 1,
        'FIM bowel management': 1,
        'FIM bed/chair/wheelchair': 1,
        'FIM toilet': 1,
        'FIM tub/shower': 1,
        'FIM stairs': 1,
        'Motor FIM admission': 1,
        'FIM comprehension': 1,
        'FIM expression': 1,
        #'FIM_problem_solving': 1,
        'FIM_social_interaction': 1,
        'FIM memory': 1,
        'industrial\xa0injury\xa0insurance': 1,
        #'Occupation': 1,
        'Housemate': 1,
        'Marital status': 1,
        "Driver\'s license": 1,
        'Cause of injury': 1,
        'Raiographic abnormality': 1,
        'Presence of OPLL/OYL': 1,
        'Surgery': 1,
        'Blood transfusion': 1,
        'Associated injury': 1,
        'Hyper tension': 1,
        'Cardiovascular disease': 1,
        'Stroke': 1,
        'Diabetes': 1,
        'Liver disease': 1,
        'Respiratory disease': 1,
        'Kidney disease': 1,
        'Right C5 motor': 1,
        'Left C5 motor': 1,
        'Right C6 motor': 1,
        'Left C6 motor': 1,
        'Right C7 motor': 1,
        'Left C7 motor': 1,
        'Right C8 motor': 1,
        'Left C8 motor': 1,
        'Right T1 motor': 1,
        'Left T1 motor': 1,
        'Right L2 motor': 1,
        'Left L2 motor': 1,
        'Right L3 motor': 1,
        'Left L3 motor': 1,
        'Right L4 motor': 1,
        'Left L4 motor': 1,
        'Right L5 motor': 1,
        'Left L5 motor': 1,
        'Right S1 motor': 1,
        'Left S1 motor': 1,
        'Sensory pin prick score': 1,
        'Sensory_score_light_touch': 1,
        'ASIA_impairment_scale_admission': 1,
        'Neurological_level_of_injury': 1
    }

    no_use_df = pd.DataFrame([no_use])

    features_df = pd.concat([features_df, no_use_df], axis=1)
    predictions_data = predict_model(model, features_df)
    predicted_class = predictions_data["Label"][0]
    score = predictions_data["Score"][0]

    res = {0: "DEPENDENT", 1: "INDEPENDENT"}
    st.title('Result: Based on feature values, the patient\'s ambulation will be '+ res[predicted_class])
    st.title('Probability: ' + str(score))


