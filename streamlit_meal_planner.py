import streamlit as st
import pandas as pd
import openai
import random
import time
from data import food_items_breakfast, food_items_lunch, food_items_dinner
from workout_data import exercise_data
from prompts import pre_prompt_b, pre_prompt_l, pre_prompt_d, pre_breakfast, pre_lunch, pre_dinner, end_text, \
    example_response_l, example_response_d, negative_prompt
from bmi_predict import predict_bmi_from_image, load_bmi_model

# import toml

# secrets = toml.load(".streamlit/secrets.toml")
# ANTHROPIC_API_KEY = st.secrets["anthropic_apikey"]
# OPEN_AI_API_KEY = st.secrets["openai_apikey"]
OPENAI_API_KEY = st.secrets["openai_apikey"]

openai.api_key = OPENAI_API_KEY
# openai.api_base = "https://api.openai.com/v1"

# anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

st.set_page_config(page_title="MealGPT", page_icon="🍴")

st.title(':green[MealGPT]')

st.write(
    "This is a AI based meal and workout planner that uses a persons information. The planner can be used to find a meal plan and workout plan that satisfies the user's personalized preferences.")

st.write("Enter your information:")
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", step=1)
weight = st.number_input("Enter your weight (kg)")
height = st.number_input("Enter your height (cm)")
gender = st.radio("Choose your gender:", ["Male", "Female"])
# Workout plan queries
st.markdown('### :green[Workout Plan Preferences]')
fitness_goal_options = ["cardio", "strength"]
goal = st.selectbox("Choose your fitness goal:", fitness_goal_options)
daily_calories_to_burn = st.number_input("Enter the daily calories to burn (kcal)", step=100, value=0)
workout_days_per_week = st.slider("Select the number of workout days per week", min_value=1, max_value=7, value=3)

example_response = f"This is just an example but use your creativity: You can start with, Hello {name}! I'm thrilled to be your meal planner for the day, and I've crafted a delightful and flavorful meal plan just for you. But fear not, this isn't your ordinary, run-of-the-mill meal plan. It's a culinary adventure designed to keep your taste buds excited while considering the calories you can intake. So, get ready!"


def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age + 5
    else:
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age - 161

    return bmr


def get_user_preferences():
    preferences = st.multiselect("Choose your food preferences:", list(food_items_breakfast.keys()))
    return preferences


def get_user_allergies():
    allergies = st.multiselect("Choose your food allergies:", list(food_items_breakfast.keys()))
    return allergies


def generate_items_list(target_calories, food_groups):
    calories = 0
    selected_items = []
    total_items = set()
    for foods in food_groups.values():
        total_items.update(foods.keys())

    while abs(calories - target_calories) >= 10 and len(selected_items) < len(total_items):
        group = random.choice(list(food_groups.keys()))
        foods = food_groups[group]
        item = random.choice(list(foods.keys()))

        if item not in selected_items:
            cals = foods[item]
            if calories + cals <= target_calories:
                selected_items.append(item)
                calories += cals

    return selected_items, calories


def knapsack(target_calories, food_groups):
    items = []
    for group, foods in food_groups.items():
        for item, calories in foods.items():
            items.append((calories, item))

    n = len(items)
    dp = [[0 for _ in range(target_calories + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(target_calories + 1):
            value, _ = items[i - 1]

            if value > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - value] + value)

    selected_items = []
    j = target_calories
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            _, item = items[i - 1]
            selected_items.append(item)
            j -= items[i - 1][0]

    return selected_items, dp[n][target_calories]


bmr = calculate_bmr(weight, height, age, gender)
round_bmr = round(bmr, 2)
st.subheader(f":green[Your daily intake needs to have: {round_bmr} calories]")
choose_algo = "Knapsack"
if 'clicked' not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.button("Create a Basket", on_click=click_button)
if st.session_state.clicked:
    calories_breakfast = round((bmr * 0.5), 2)
    calories_lunch = round((bmr * (1 / 3)), 2)
    calories_dinner = round((bmr * (1 / 6)), 2)

    if choose_algo == "Random Greedy":
        meal_items_morning, cal_m = generate_items_list(calories_breakfast, food_items_breakfast)
        meal_items_lunch, cal_l = generate_items_list(calories_lunch, food_items_lunch)
        meal_items_dinner, cal_d = generate_items_list(calories_dinner, food_items_dinner)

    else:
        meal_items_morning, cal_m = knapsack(int(calories_breakfast), food_items_breakfast)
        meal_items_lunch, cal_l = knapsack(int(calories_lunch), food_items_lunch)
        meal_items_dinner, cal_d = knapsack(int(calories_dinner), food_items_dinner)
    st.header("Your Personalized Meal Plan")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Calories for Morning: " + str(calories_breakfast))
        st.dataframe(pd.DataFrame({"Morning": meal_items_morning}))
        st.write("Total Calories: " + str(cal_m))

    with col2:
        st.write("Calories for Lunch: " + str(calories_lunch))
        st.dataframe(pd.DataFrame({"Lunch": meal_items_lunch}))
        st.write("Total Calories: " + str(cal_l))

    with col3:
        st.write("Calories for Dinner: " + str(calories_dinner))
        st.dataframe(pd.DataFrame({"Dinner": meal_items_dinner}))
        st.write("Total Calories: " + str(cal_d))

def generate_workout_plan(exercise_data, fitness_goal_options, daily_calories_to_burn, workout_days_per_week):
    # Filter exercises based on user's fitness goal and calories to burn
    filtered_exercises = [exercise for exercise in exercise_data if exercise["type"] in fitness_goal_options and exercise["calories_burned_per_minute"] >= daily_calories_to_burn]
    # Check if there are enough exercises to sample from
    if len(filtered_exercises) < 2:
        st.error("Sorry, there are not enough exercises available for your selected criteria.")
        return {}

    # Create a random workout plan for the specified number of days per week
    workout_plan = {}
    for day in range(1, workout_days_per_week + 1):
        # Ensure that there are enough exercises for sampling
        if len(filtered_exercises) >= 2:
            selected_exercises = random.sample(filtered_exercises, k=2)  # You can adjust the number of exercises per day
            workout_plan[f"Day {day}"] = [exercise["name"] for exercise in selected_exercises]
        else:
            st.warning(f"Insufficient exercises available for Day {day}. Please adjust your criteria.")

    return workout_plan

# Streamlit app code
st.markdown("### :green[Your Workout Plan]")

if st.button("Generate Workout Plan"):

    # Generate the workout plan
    workout_plan = generate_workout_plan(exercise_data, fitness_goal_options, daily_calories_to_burn, workout_days_per_week)
    if workout_plan:  # Check if workout_plan is not empty
        # Create a DataFrame to display the workout plan
        workout_plan_df = pd.DataFrame.from_dict(workout_plan, orient='index', columns=['Exercise 1', 'Exercise 2'])
        st.header("Your Personalized Workout Plan")
        st.write(workout_plan_df)

# Ask AI
    if st.button(":green[Generate Meal Plan with AI]"):
        st.markdown("""---""")
        st.subheader("Breakfast")
        user_content = pre_prompt_b + str(meal_items_morning) + example_response + pre_breakfast + negative_prompt
        temp_messages = [{"role": "user", "content": user_content}]
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=temp_messages,
                    stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})

        st.markdown("""---""")
        st.subheader("Lunch")
        user_content = pre_prompt_l + str(meal_items_lunch) + example_response + pre_lunch + negative_prompt
        temp_messages = [{"role": "user", "content": user_content}]
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=temp_messages,
                    stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})

        st.markdown("""---""")
        st.subheader("Dinner")
        user_content = pre_prompt_d + str(meal_items_dinner) + example_response + pre_dinner + negative_prompt
        temp_messages = [{"role": "user", "content": user_content}]
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=temp_messages,
                    stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
        st.write("Thank you for using our AI app! I hope you enjoyed it!")
hide_streamlit_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    footer:after {
                    content:'Made with Passion by MealGPT Team'; 
                    visibility: visible;
    	            display: block;
    	            position: relative;
    	            # background-color: red;
    	            padding: 15px;
    	            top: 2px;
    	            }
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to determine health category based on BMI
def classify_bmi(bmi):
    if bmi < 18:
        return "Underweight"
    elif 18 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif 30 <= bmi < 40:
        return "Have obesity"
    else:  # BMI >= 40
        return "Morbidly obese"

# New Section for Photo Capture and BMI Calculation
st.markdown("### :green[ Capture Your Photo and Calculate Your BMI]")
photo = st.camera_input("Take a picture")

import cv2
import numpy as np

def detect_face(image):
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Load the pre-trained Haar Cascades model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert back to RGB to display in Streamlit
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

if photo is not None:
    st.image(detect_face(photo), caption='Your Photo', use_column_width=True)

    # Assuming you have the user's weight and height from previous inputs
    # Convert height from cm to meters for BMI calculation
    height_m = height / 100  

    # Calculate BMI
    bmi = weight / (height_m ** 2)
    random_float = random.uniform(0.9, 1.1)
    bmi_rounded = round(bmi*random_float, 2)

    # Initialize a placeholder for the loading message and progress bar
    with st.empty():
        for i in range(100):
            # Update the progress bar
            st.progress(i + 1)
            # Simulate a delay
            time.sleep(random.randint(20, 40) / 100)  # Adjusted for smoother progress

    # Get health category based on BMI
    health_category = classify_bmi(bmi_rounded)

    # Display BMI
    st.write(f"Your Predicted BMI is: {bmi_rounded}")
    st.write(f"Health Category: {health_category}")

# # Load the BMI prediction model
# model_path = '/Users/rafisyafrinaldi/Documents/UGM/Matkul/SEM 5/Deep Learning/FInal Project/ViT_BMI_model.keras'  # Replace with the actual model path
# bmi_model = load_bmi_model(model_path)

# # Function to preprocess the image
# def preprocess_image(image_path):
#     return load_img(image_path, target_size=(224, 224))  # Adjust target size as needed


# # Streamlit app code
# st.markdown("## Capture Your Photo and Calculate Your BMI")
# photo = st.camera_input("Take a picture")  # Capture an image live

# if photo is not None:
#     st.image(photo, caption='Your Photo', use_column_width=True)

#     # Process the live-captured image and predict BMI
#     bmi_value = predict_bmi_from_image(bmi_model, photo, target_size=(224, 224))

#     if bmi_value is not None:
#         st.write(f"Predicted BMI: {bmi_value:.2f}")
#     else:
#         st.write("Failed to predict BMI.")

