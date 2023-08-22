#Packages Required
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from streamlit_lottie import st_lottie
import plotly_express as px

#Setting WebPage Configurations
st.set_page_config(page_title="Covid-19 Prediction", page_icon=":notebook:", layout="wide",initial_sidebar_state="expanded")

#Function to Load Animations from Lottie Website
def load_lottie(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

#URLs of the Lottie Animations
lottie1 = load_lottie("https://assets10.lottiefiles.com/packages/lf20_66hoasot.json")
lottie2 = load_lottie("https://assets6.lottiefiles.com/packages/lf20_wrb0q9s6.json")

#Creating Dictionaries List
gender_dict = {"Female":1,"Male":2}
features_dict = {"Yes":1,"No":2}
pt_dict = {"Returned Home":1,"Hospitalized":2}

#Creating Functions to use the Dictionaries
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
def get_key(val):
    features_dict = {"Yes":1,"No":2}
    for key, value in features_dict.items():
        if val == key:
            return key
def get_feature_value(val):
    features_dict = {"Yes":1,"No":2}
    for key, value in features_dict.items():
        if val == key:
            return value

# List of colums in the Dataset
plotting = ["CLASSIFICATION_FINAL", "GENDER", "PATIENT_TYPE", "INTUBED", "PNEUMONIA", "AGE",
    "PREGNANT", "DIABETES", "CHRONIC_DISEASE", "ASTHMA", "IMMUNOSUPPRESSED", 
    "HYPERTENSION", "CARDIOVASCULAR",  "OBESITY", "TOBACCO", "ICU"]

# Main Function
def main():
    """Prediction App"""
    st.title("Covid-19 Prediction App")

    # Creating the Menu Tab
    choice = option_menu(
            menu_title=None,
            options=["Home", "Data", "Analyzation", "Prediction"],
            icons=["house","bar-chart","activity","graph-up-arrow"],
            orientation="horizontal")

    # Designing the Home Tab
    if choice == "Home":
        with st.container():
            # Splitting the Tab with 2:1 ratio
            left_column, right_column = st.columns((2,1))
            with left_column:
                st.header("Home")
                st.subheader("What Is Covid?")
                st.write("Coronavirus 2019 (COVID-19) is a contagious disease caused by a virus, the Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2). The disease quickly spread worldwide, resulting in the COVID-19 pandemic.")
                st.write("COVID-19 transmits when people breathe air contaminated by droplets and small airborne particles containing the virus. The risk of breathing these is highest when people are in close proximity, but they can be inhaled over longer distances, particularly indoors. Transmission can also occur if contaminated fluids are splashed or sprayed in the eyes, nose, or mouth, or, more rarely, via contaminated surfaces. People remain contagious for up to 20 days and can spread the virus even if they do not develop symptoms.")
        
            with right_column:
                st.write("ㅤ")
                st.write("ㅤ")
                # Calling the Lottie Function to load the annimation 
                st_lottie(lottie1,height=300,key="Corona")

            with st.container():
                left_column2, right_column2 = st.columns((1,2))
                with left_column2:
                    st_lottie(lottie2,height=300,key="Corona2")
        
                with right_column2:
                    st.write("ㅤ")
                    st.write("The complex and highly contagious nature of the COVID-19 had led the WHO to declare this outbreak a public health emergency, which consequently brought significant health, economic, and social challenges. It is mandatory to be able to differentiate between COVID-19 from other pneumonia-like diseases early after symptom development. Due to the high-level spread and increasing epidemiology trend of COVID-19, its early diagnosis and rapid isolation of infected people play a key role in confining this virus and thereby reducing the disease outbreak and mortality rate.")
                    st.write("Covid -19 is a new infection that’s been threat to the human life. Coronavirus are in the range of 80- 160 nanometers size. Very basic signs of Covid-19 are high temperature and Breathing faster than usual. Symptoms of Covid-19 are Fatigue, Nausea, Loss of taste or smell, Muscle Ache, Cough, Headache, Sore Throat, Tiredness, chills and Fever. There are some cases where people don’t show any symptoms and are called asymptomatic.")
        
    # Designing the Data Tab
    elif choice == "Data":
        st.subheader("Overview Of The Data :")

        # Viewing a outline of the data
        data = pd.read_csv("Covid_Data.csv")
        st.write(data)
        st.write("The dataset used for this process has data of 10,48,576 people, in which each individual has 15 characteristics, and the final classification of whether the person has Corona or not.")
        st.write("The Final Classification has a Range of 1 to 7, where 1 is very serious condition for the person and 7 is completely free from Corona.")
        st.write("The Age of the person ranges from 0 to 121, Gender is represented by 1-Female and 2-Male, and other characteristics is represented by 1-Yes and 2-No.")
        with st.container():
            left_column, middle_column, right_column = st.columns((0.75,1,1))
            with middle_column:
                st.write("ㅤ")

                # Creating a Pie Chart based on the choosen option
                value = st.selectbox("Choose the Parameter to Plot:",plotting)
                pie_chart = px.pie(data,title='Pie Chart :', values= 'CLASSIFICATION_FINAL', names=value)
                st.plotly_chart(pie_chart)
            if value == 'GENDER':
                st.write("Gender of the Patient : 1 - Female , 2 - Male")
            elif value == 'PATIENT_TYPE':
                st.write("Type of the Patient : 1 - Returned Home , 2 - Hospitalized")
            elif value == 'AGE':
                st.write("ㅤ")
            elif value == 'CLASSIFICATION_FINAL':
                st.write("Final Classification of the Patient : 1 - Intensive state of Corona , 2 - Highly Exposed to Corona , 3 - Affected by Corona , 4 - May have Corona , 5 - Mildly Safe , 6 - Safe from Corona, 7 - Completely Safe from Corona")
            else :
                st.write("1 - Yes , 2 - No")

    # Designing the Analyzation Tab
    elif choice == "Analyzation":
        st.subheader("Analyzation Of The Data :")
        st.subheader("Data Visualization Plotting")
        df = pd.read_csv("data/Covid_Data.csv")
        st.write(plotting)
        parameter = st.selectbox("Choose the Parameter to Plot :",plotting)

        # Plotting a Bar Chart based on a given option
        df[parameter].value_counts().plot(kind='bar',)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        with st.container():
            left_column, middle_column, right_column = st.columns((0.75,1,1))
            with middle_column:
                st.pyplot()
            
    # Designing the Prediction Tab
    elif choice == "Prediction":
            st.subheader("Predictive Analytics")
            df = pd.read_csv("data/Covid_Data.csv")
            X = df[["GENDER", "PATIENT_TYPE", "INTUBED", "PNEUMONIA", "AGE",
            "PREGNANT", "DIABETES", "CHRONIC_DISEASE", "ASTHMA", "IMMUNOSUPPRESSED", 
            "HYPERTENSION", "CARDIOVASCULAR",  "OBESITY", "TOBACCO", "ICU"]]
            y = df["CLASSIFICATION_FINAL"]

            # Splitting the Train and Test Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Getting the input data in form of KEYS from the user
            gender = st.selectbox("Gender of the patient",tuple(gender_dict.keys()))
            patient_type = st.selectbox("Type Of patient",tuple(pt_dict.keys()))
            intubed = st.selectbox("Is the patient kept in Ventilator",tuple(features_dict.keys()))
            pneumonia = st.selectbox("Do the patient have Pneumonia",tuple(features_dict.keys()))
            age = st.number_input("Age",0,121)
            pregnant = st.selectbox("Do the patient is Pregnant",tuple(features_dict.keys()))
            diabetes = st.selectbox("Do the patient have Diabetes",tuple(features_dict.keys()))
            chronic = st.selectbox("Do the patient have any Chronic Disease",tuple(features_dict.keys()))
            asthma = st.selectbox("Do the patient have Asthma",tuple(features_dict.keys()))
            immunosupp = st.selectbox("Is the patient ImmunoSuppressed",tuple(features_dict.keys()))
            hypertens = st.selectbox("Do the patient suffer from HyperTension",tuple(features_dict.keys()))
            cardiovas = st.selectbox("Do the patient have any CardioVascular Disease",tuple(features_dict.keys()))
            obesity = st.selectbox("Do the patient have Obesity",tuple(features_dict.keys()))
            tobacco = st.selectbox("Do the patient consume Tobacco Products",tuple(features_dict.keys()))
            icu = st.selectbox("Do the patient admitted in ICU",tuple(features_dict.keys()))

            # Converting the input KEY into Values and sent to the ML Model        
            features_list = [get_value(gender,gender_dict),get_value(patient_type,pt_dict),get_value(intubed,features_dict),
            get_value(pneumonia,features_dict),age,get_value(pregnant,features_dict),get_value(diabetes,features_dict),
            get_value(chronic,features_dict),get_value(asthma,features_dict),get_value(immunosupp,features_dict),
            get_value(hypertens,features_dict),get_value(cardiovas,features_dict),get_value(obesity,features_dict),
            get_value(tobacco,features_dict),get_value(icu,features_dict)]

            if st.button("Submit"):
                    # Creating the ML Model using Logistic Regression
                    clf = LogisticRegression() 
                    clf.fit(X, y)
                    joblib.dump(clf, "clf.pkl")
                    clf = joblib.load("clf.pkl")
                    X = pd.DataFrame([features_list], 
                    columns = ["GENDER", "PATIENT_TYPE", "INTUBED", "PNEUMONIA", "AGE",
    "PREGNANT", "DIABETES", "CHRONIC_DISEASE", "ASTHMA", "IMMUNOSUPPRESSED", 
    "HYPERTENSION", "CARDIOVASCULAR",  "OBESITY", "TOBACCO", "ICU"])

                    # Predicting the output based on the input parameters given
                    prediction = clf.predict(X)[0]

                    # Printing the Final Result Based on the Prediction
                    st.write("Based on the given parameters, the Final Classification Is :")
                    if prediction == 1:
                        st.info("The Person is severely affected by Corona and need to Hospitaziled")
                    elif prediction == 2:
                        st.info("The Person is exposed to Corona and must consult a Doctor")
                    elif prediction == 3:
                        st.info("The Person has Corona and need to be isolated in home")
                    elif prediction == 4:
                        st.info("The Person may have Corona and need to be checked")
                    elif prediction == 5:
                        st.info("The Person may suffer from fever and may have Corona")
                    elif prediction == 6:
                        st.info("The Person may have cough and cold, but not Corona")
                    elif prediction == 7:
                        st.info("The Person is completely safe and not affected by Corona")


if __name__ == '__main__':
    main()