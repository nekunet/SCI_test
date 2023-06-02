import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model


@st.cache(allow_output_mutation=True)
def load_model_for_cache2():
    ret = load_model('finalized_regression_model')
    return ret    


st.title('Prediction for functional ambulation')
st.write('This is a web app for predicting the total motor Functional Independence Measure (FIM) score in patients with spinal cord injury at discharge from a rehabilitation hospital.')

st.write('The default value of each feature in the web application is a placeholder. Users must modify their inputs according to the clinical characteristics of each individual patient. Users should be aware that the algorithm requires complete information about the items contained in the interface to run the algorithm. Once the input is completed, users can click the “Predict” button at the bottom to see the prediction. The app will tell users the patient’s estimated total motor FIM score.')

st.write('PLEASE NOTE: Hitting the + button or - button repeatedly, the app may crash.')
#dataset.csvの特徴量の変更する


#Age = st.sidebar.slider(label = 'Age', min_value = 18,
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

FIM_eating = st.sidebar.number_input(label = 'FIM eating (1-7)',
                          min_value = 1,
                          max_value = 7,
                          value = 4,
                          step = 1)

# FIM_grooming = st.sidebar.number_input(label = 'FIM grooming (1-7)',
#                           min_value = 1,
#                           max_value = 7,
#                           value = 3,
#                           step = 1)

#FIM_dressing_upper_body = st.sidebar.number_input(label = 'FIM dressing upper body (1-7)',
 #                         min_value = 1,
  #                        max_value = 7,
  #                        value = 2,
  #                        step = 1 )

# FIM_dressing_lower_body = st.sidebar.number_input(label = 'FIM dressing lower body (1-7)',
#                           min_value = 1,
#                           max_value = 7,
#                           value = 2,
#                           step = 1 )

# FIM_bladder_management = st.sidebar.number_input(label = 'FIM_bladder_management (1-7)',
#                           min_value = 1,
#                           max_value = 7,
#                           value = 2,
#                           step = 1 )

#FIM_walk_wheelchair = st.sidebar.number_input(label = 'FIM walk/wheelchair (1-7)',
 #                         min_value = 1,
   #                       max_value = 7,
#                          value = 2,
#                          step = 1 )


FIM_problem_solving = st.sidebar.number_input(label = 'FIM problem solving (1-7)',
                          min_value = 1,
                          max_value = 7,
                          value = 7,
                          step = 1)

# FIM_memory = st.sidebar.number_input(label = 'FIM memory (1-7)',
#                           min_value = 1,
#                           max_value = 7,
#                           value = 7,
#                           step = 1)

Total_FIM_score = st.sidebar.number_input(label = 'Total FIM score (18-126)', min_value = 18,
                          max_value = 126,
                          value = 63,
                          step = 1)

Right_C5_motor = st.sidebar.number_input(label = 'Right C5 motor (0-5)',
                          min_value = 0,
                          max_value = 5,
                          value = 4,
                          step = 1)

Right_C7_motor = st.sidebar.number_input(label = 'Right C7 motor (0-5)',
                          min_value = 0,
                          max_value = 5,
                          value = 4,
                          step = 1)

# Left_C7_motor = st.sidebar.number_input(label = 'Left C7 motor (0-5)',
#                           min_value = 0,
#                           max_value = 5,
#                           value = 4,
#                           step = 1)

# Left_C8_motor = st.sidebar.number_input(label = 'Left C8 motor (0-5)',
#                           min_value = 0,
#                           max_value = 5,
#                           value = 4,
#                           step = 1)

Right_L2_motor = st.sidebar.number_input(label = 'Right L2 motor (0-5)',
                          min_value = 0,
                          max_value = 5,
                          value = 2,
                          step = 1)

# Left_L4_motor = st.sidebar.number_input(label = 'Left L4 motor (0-5)',
#                           min_value = 0,
#                           max_value = 5,
#                           value = 2,
#                           step = 1)

ASIA_motor_score = st.sidebar.number_input(label = 'ASIA motor score (0-100)', min_value = 0,
                          max_value = 100 ,
                          value = 60,
                          step = 1)

Sensory_pin_prick_score = st.sidebar.number_input(label = 'Sensory pin prick score (0-112)', min_value = 0,#
                          max_value = 112 ,
                          value = 60,
                          step = 1)

# ASIA_impairment_scale =  st.sidebar.selectbox('ASIA impairment scale',
#                           ("A", "B", "C", "D"))

# ASIA_impairment_scale_dic = {"A": 1, "B": 2, "C": 3, "D": 4}

Educational_background = st.sidebar.selectbox('Educational background (final education)',
                          ("Elementary school", "Junior high school", "High school", "University", "Graduate school"))

Educational_background_dic = {"Elementary school": 2, "Junior high school": 3, "High school": 4, "University": 5, "Graduate school": 6}

Neurological_level_of_injury = st.sidebar.selectbox('Neurological level of injury',
                   ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "T1", "T2", "T3",
                    "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1",
                    "L2", "L3", "L4", "L5", "S1", "S2", "S3", "S4"))

NLI_value_dic = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8,
                 "T1": 9, "T2": 10, "T3": 11, "T4": 12, "T5": 13, "T6": 14, "T7": 15,
                 "T8": 16, "T9": 17, "T10": 18, "T11": 19, "T12": 20, "L1": 21, "L2": 22,
                 "L3": 23, "L4": 24, "L5": 25, "S1": 26, "S2": 27, "S3": 28, "S4": 29}


if st.button('Predict'):
    model = load_model_for_cache2()

    features = {'Age': Age,
            'Days_from_injury_to_admission': Days_from_injury_to_admission,
            'FIM_eating': FIM_eating,
            #'FIM_grooming': FIM_grooming,
            #'FIM_dressing_lower_body': FIM_dressing_lower_body,
            #'FIM_dressing_upper_body': FIM_dressing_upper_body,
            #'FIM_bladder_management': FIM_bladder_management,
            #'FIM_social_interaction': FIM_social_interaction,
            #'FIM_walk_wheelchair': FIM_walk_wheelchair,
            'FIM_problem_solving': FIM_problem_solving,
            #'FIM_memory': FIM_memory,
            'Total_FIM_score': Total_FIM_score,
            'Educational_background': Educational_background_dic[Educational_background],
            'Neurological_level_of_injury': NLI_value_dic[Neurological_level_of_injury],
            'ASIA_motor_score': ASIA_motor_score,
            'Right_L2_motor': Right_L2_motor,
            #'Left_L4_motor': Left_L4_motor,
            'Right_C5_motor': Right_C5_motor,
            'Right_C7_motor': Right_C7_motor,
            #'Left_C7_motor':  Left_C7_motor,
            #'Left_C8_motor': Left_C8_motor,
            'Sensory_pin_prick_score': Sensory_pin_prick_score,
            #'ASIA_impairment_scale': ASIA_impairment_scale_dic[ASIA_impairment_scale],
            }


    features_df  = pd.DataFrame([features])

    no_use = {
        'Sex': 1,
        'ASIA_impairment_scale': 1,

        'FIM_bathing': 1,
        'FIM_walk_wheelchair': 1,
        'FIM_toileting': 1,
        'FIM_dressing_upper_body': 1,
        'FIM_bowel_management': 1,
        'FIM_bed_chair_wheelchair': 1,
        'FIM_toilet': 1,
        'FIM_tub_shower': 1,
        'FIM_stairs': 1,
        'Motor_FIM_admission': 1,
        'FIM_comprehension': 1,
        'FIM_expression': 1,
        'FIM_grooming': 1,
        'FIM_dressing_lower_body': 1,
        'FIM_bladder_management': 1,
        'FIM_memory': 1,
        'FIM_social_interaction': 1,
        'industrial_injury_insurance': 1,
        'Occupation': 1,
        'Housemate': 1,
        'Marital_status': 1,
        "Driver\'s_license": 1,
        'Cause_of_injury': 1,
        'Radiographic_abnormality': 1,
        'Presence_of_OPLL_OYL': 1,
        'Surgery': 1,
        'Blood_transfusion': 1,
        'Associated_injury': 1,
        'Hyper_tension': 1,
        'Cardiovascular_disease': 1,
        'Stroke': 1,
        'Diabetes': 1,
        'Liver_disease': 1,
        'Respiratory_disease': 1,
        'Kidney_disease': 1,
        #'Right_C5_motor':1,
        'Left_C5_motor':1,
        'Right_C6_motor':1,
        'Left_C6_motor':1,
        #'Right_C7_motor': 1,
        'Left_C7_motor': 1,
        'Right_C8_motor': 1,
        'Left_C8_motor': 1,
        'Right_T1_motor': 1,
        'Left_T1_motor':1,
        #'Right_L2_motor': 1,
        'Left_L2_motor': 1,
        'Right_L3_motor': 1,
        'Left_L3_motor': 1,
        'Right_L4_motor': 1,
        'Left_L4_motor': 1,
        'Right_L5_motor': 1,
        'Left_L5_motor': 1,
        'Right_S1_motor': 1,
        'Left_S1_motor': 1,
        'Sensory_light_touch_score': 60,
    }

    no_use_df = pd.DataFrame([no_use])

    features_df = pd.concat([features_df, no_use_df], axis=1)
    predictions_data = predict_model(model, features_df)
    predicted_class = predictions_data["Label"][0]
    #score = predictions_data["Score"][0]
    y = round(predicted_class, 1)
    #a = y.quantize(Decimal('0.1', rounding=ROUND_HALF_UP))

    #res = {0: "DEPENDENT", 1: "INDEPENDENT"}
    #st.title('Result: Based on feature values, the patient\'s total motor FIM score will be '+ str(predicted_class))
    st.title('Result: Based on feature values, the patient\'s total motor FIM score will be '+ str(y))
    #st.title('Probability: ' + str(score))
