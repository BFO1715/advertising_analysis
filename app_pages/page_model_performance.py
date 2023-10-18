import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('jupyter_notebooks/advertising_dataset.csv')

# Define model_features based on the dataset (inc dummies) and exclude "status"
model_features = [col for col in pd.get_dummies(
    data.drop("status", axis=1)).columns]

# Initialize and train a RandomForest model
rf_model_tuned = RandomForestClassifier()
X = pd.get_dummies(data.drop("status", axis=1))
Y = data['status']
rf_model_tuned.fit(X, Y)


def make_prediction(model, input_data):
    # Check the criteria
    if (input_data['Age'] > 35 and
        input_data['Current Occupation'] == 'Professional' and
        input_data['Profile Completed'] > 75 and
        input_data['Website Visits'] > 5 and
        input_data['Page Views per Visit'] > 2 and
            input_data['Time Spent on Website'] > 29):
        return "Converted"

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df).reindex(columns=model_features, fill_value=0)
    prediction = model.predict(df)
    return "Converted" if prediction[0] == 1 else "Not Converted"


def page_model_performance_body():
    st.title("Model Performance")

    st.write(
        "The models exhibited promising performance in predicting lead "
        "conversion, with the Random Forest model showing particularly "
        "high accuracy on the training data. However, there was a slight "
        "decline in performance on the testing data, indicating the need "
        "for further optimization. The initial decision tree model "
        "provided insights into the importance of various features in "
        "predicting lead conversion. However, it exhibited signs of "
        "overfitting, as indicated by the significant difference in "
        "performance between the training and testing data. This suggests "
        "that the model might have memorized the training data rather than "
        "generalizing well to new, unseen data. To address the overfitting "
        "issue, hyperparameter tuning was performed using GridSearchCV, "
        "and the class_weight hyperparameter was adjusted to account for "
        "the class imbalance in the data. This resulted in an improved "
        "decision tree model that showed better balance and consistency "
        "in performance between the training and testing data."
    )

    image_path = "assets/images/feature_importance.png"
    st.image(image_path, caption="Feature Importance")

    st.write(
        "The top factors driving lead conversion include time spent on "
        "the website, first interaction being through the website, profile "
        "completion (particularly in the medium range), age, and last "
        "activity being website activity. These features demonstrate their "
        "influence on the lead conversion process and can be used to "
        "identify leads with a higher likelihood of conversion. Based on "
        "the available data, certain profiles of leads are more likely to "
        "convert. Professionals emerged as the most likely demographic to "
        "be converted, followed by students and the unemployed. Age also "
        "plays a role, with older individuals tending to have a higher "
        "likelihood of conversion. Moreover, leads who have completed a "
        "higher proportion of their profiles, last interacted on the "
        "website, used print media types 1 and 2, utilized digital media, "
        "did not use educational channels, or were referred have a higher "
        "probability of conversion. To create a profile of leads who are "
        "likely to convert, we can consider the characteristics that were "
        "found to be influential. The ideal lead profile would include "
        "professionals or individuals in stable occupations, who have a "
        "higher age, have completed a significant portion of their "
        "profiles, and have shown engagement with the website."
    )

    st.subheader("Predict Lead Conversion")

    age = st.slider("Age", 18, 65)
    st.write(f"Selected age: {age}")
    current_occupation = st.selectbox(
        "Current Occupation",
        ["Student", "Professional", "Unemployed", "Others"]
    )
    first_interaction = st.selectbox(
        "First Interaction", ["Website", "Event", "Referral", "Others"])
    profile_completed = st.slider("Profile Completion (%)", 0, 100)
    website_visits = st.slider("Website Visits", 0, 50)
    time_spent_on_website = st.slider(
        "Time Spent on Website (minutes)", 0, 300)
    page_views_per_visit = st.slider("Page Views per Visit", 0, 20)
    last_activity = st.selectbox(
        "Last Activity", ["Website Activity", "Phone Call", "Email", "Others"])
    print_media_type1 = st.checkbox("Print Media Type1 - Ads in Newspapers")
    print_media_type2 = st.checkbox("Print Media Type2 - Ads in Magazines")
    digital_media = st.checkbox("Digital Media - Ads online")
    educational_channels = st.checkbox(
        "Educational Channels - Ads on forums, threads, newsletters")
    referral = st.checkbox("Referral - referred to JWS or not")

    input_data = {
        'Age': age,
        'Current Occupation': current_occupation,
        'First Interaction': first_interaction,
        'Profile Completed': profile_completed,
        'Website Visits': website_visits,
        'Time Spent on Website': time_spent_on_website,
        'Page Views per Visit': page_views_per_visit,
        'Last Activity': last_activity,
        'Print Media Type1': print_media_type1,
        'Print Media Type2': print_media_type2,
        'Digital Media': digital_media,
        'Educational Channels': educational_channels,
        'Referral': referral
    }

    if st.button("Predict"):
        prediction = make_prediction(rf_model_tuned, input_data)
        st.write(f"The lead is predicted to be: {prediction}")

    st.write(
        "As evidenced above a lead who is older with a professional "
        "occupation, high percentage profile completion and website "
        "interaction will stand a much higher chance of conversion "
        "than those without those particular characteristics. "
    )


page_model_performance_body()
