import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import logging
import pygsheets
import ast
import hmac
import hashlib
import math
import plotly.express as px
from datetime import datetime


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
log = logging.getLogger(__name__)
logging.basicConfig(format=FORMAT)
log.setLevel(logging.DEBUG)

if 'username_hash' not in st.session_state:
    st.session_state.username_hash = None


def periodization_equation(step, nTotalSteps, base=0.6, nCycles=3.0):
    percent = 0.0
    rate = 0.3
    x = step

    maxSteps = float(nTotalSteps) - 1.0

    b = ((float(maxSteps) + 1) / nCycles)
    period = (math.fmod(step, b) * b / (b - 1.0)) / b
    percent = ((rate / nTotalSteps) * x + base) + 0.1 * period

    return percent

def hash_username(username):
    # Ensure the username is encoded to bytes, required for hashing
    username_bytes = username.encode('utf-8')

    # Create a SHA-256 hash object
    sha256 = hashlib.sha256()

    # Update the hash object with the username bytes
    sha256.update(username_bytes)

    return sha256.hexdigest()


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        st.header("User Login")
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            username_hash = hash_username(st.session_state["username"])
            st.session_state['username_hash'] = username_hash
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


def create_new_workout():
    global start_date, num_weeks

    # Define the mapping dictionary
    day_to_number = {
        'Sunday': 0,
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6
    }

    col_a, col_b, col_c = st.columns(3)
    with col_b:
        days_of_week_options = ['Sunday',
                                'Monday',
                                'Tuesday',
                                'Wednesday',
                                'Thursday',
                                'Friday', 'Saturday',
                                ]
        days_of_week = st.multiselect('Select Days of the Week',
                                      default=['Monday', 'Wednesday', 'Friday'],
                                      options=days_of_week_options)
    with col_a:
        start_date = st.date_input('Select workout start date')
    with col_c:
        num_weeks = st.number_input("Number of weeks",
                                    min_value=4,
                                    max_value=16,
                                    value=12)

    # Convert to list of numbers using list comprehension
    day_numbers = [day_to_number[day] for day in days_of_week]


def expand_description(row):
    description = row.get('Description')
    timeframe = row.get('Workout Weeks')
    return f"{description} - {timeframe} Weeks"

def extract_workout_day_id(row):
    days_per_week = row.get('Days Per Week')
    index_val = row.name
    return index_val % days_per_week

def generate_workout(row, df_workout_details):
    _df = pd.DataFrame(row).T
    return _df.merge(df_workout_details, on=['Workout-ID', 'Workout-Day-ID'])


def generate_workout_details(row, df_excercises, df_user_logs):
    progression = row.get('Progression')
    exercise = row.get('Exercise')
    exercise_type = row.get('Type')
    flag_random = row.get('Random')
    sets = row.get('Sets')
    repititions = row.get('Repetitions')
    time_val = row.get('Time')
    workout_date = row.get('Workout Date')
    start_date = row.get('Start Date')
    workout_id = row.get('Workout-ID')
    workout_day_id = row.name
    cycles = row.get('Cycles')
    weeks = row.get('Weeks')
    days_per_week = row.get('Days Per Week')
    num_workouts = weeks * days_per_week

    _df_excercise = df_excercises[df_excercises['Exercise'] == exercise]
    if flag_random == True:
        _df_excercise = df_excercises[df_excercises[exercise_type] == True].sample(n=1)

    excercise_val = _df_excercise.iloc[0]['Exercise']
    units_val = _df_excercise.iloc[0]['Units']

    df_filtered = df_user_logs[df_user_logs['exercise'] == excercise_val]
    ratio_val = 1.0
    if df_filtered.empty:
        ratio_val = _df_excercise.iloc[0]['CNJ Ratio']
        df_filtered = df_user_logs[df_user_logs['exercise'] == 'BodyWeight']

    df_filtered['date'] = pd.to_datetime(df_filtered['date'], format='%m/%d/%Y')
    df_sorted = df_filtered.sort_values(by='date', ascending=False)
    target_weight = df_sorted.iloc[0]['value']

    exercise_defintions = []
    for set in range(sets):

        if progression == 'Periodization':
            periodization_val = periodization_equation(step=workout_day_id,
                                                        nTotalSteps=num_workouts,
                                                        base=0.6,
                                                        nCycles=cycles)
            weight_val = target_weight * ratio_val * periodization_val
            repititions_val = repititions
        elif progression == 'Flat':
            weight_val = target_weight * ratio_val
            repititions_val = repititions
        else:
            weight_val = target_weight * ratio_val
            repititions_val = repititions

        if exercise_type != 'Core':
            weight_val = round(weight_val / 2.5) * 2.5
            weight_val = max(weight_val, 40.0)
        else:
            weight_val = 0.0

        _exercise = {
            'Workout-ID': workout_id,
            'Workout-Day-ID': workout_day_id,
            'Workout Date': workout_date,
            'Workout Start Date': start_date,
            'Excercise-Set-ID': set,
            'Exercise-Type': exercise_type,
            'Exercise': excercise_val,
            'CNJ Raio': ratio_val,
            'Repetitions':repititions_val,
            'Weight': weight_val,
            'Units': units_val,
            'Random Flag': flag_random,
        }

        exercise_defintions.append(_exercise)

    return pd.DataFrame(exercise_defintions)


def app():

    if not check_password():
        st.stop()

    st.title("🎈 Workout Planner")

    df_users, df_excercises, df_workouts, df_workout_details, df_user_logs = gather_data()

    flag_create_new_workout = st.checkbox("Create New Workout")
    if flag_create_new_workout:
        create_new_workout()

    users = df_users['username_hash'].unique()
    if st.session_state.username_hash in users:
        st.subheader('Available Workouts')

        df_user_workouts = df_users.merge(df_workouts, on=['Workout-ID'], how='left')
        df_user_workouts['DescriptionTime'] = df_user_workouts.apply(lambda row: expand_description(row), axis=1)
        df_user_workouts['Start Date'] = pd.to_datetime(df_user_workouts['Start Date'])
        df_user_specific_logs = df_user_logs[df_user_logs['username_hash'] == st.session_state.username_hash]

        # Select the workout from the user
        workout_choice = st.selectbox('Workout Selection', options=df_user_workouts['DescriptionTime'].unique())

        df = df_user_workouts[df_user_workouts['DescriptionTime'] == workout_choice]
        workout_ids = df['Workout-ID'].unique()
        for workout_id in workout_ids:
            st.subheader(f'Workout: ')
            start_date = df.loc[df['Workout-ID'] == workout_id, 'Start Date'].iloc[0]
            days_of_week = df.loc[df['Workout-ID'] == workout_id, 'Days of Week'].iloc[0]
            days_of_week = ast.literal_eval(days_of_week)
            num_weeks = df.loc[df['Workout-ID'] == workout_id, 'Workout Weeks'].iloc[0]
            days_per_week = df.loc[df['Workout-ID'] == workout_id, 'DaysPerWeek'].iloc[0]
            num_cycles = df.loc[df['Workout-ID'] == workout_id, 'Cycles'].iloc[0]

            # Generate the range of dates
            dates = pd.date_range(start=start_date, periods=num_weeks * 3, freq='B').to_series()

            # Filter to only include Mondays, Wednesdays, and Fridays
            dates = dates[dates.dt.dayofweek.isin(days_of_week)]

            df_user_workout = pd.DataFrame(dates)
            df_user_workout.reset_index(names='Workout Date', inplace=True)
            df_user_workout.drop(columns=0, inplace=True)
            df_user_workout['Workout-ID'] = workout_id
            df_user_workout['Start Date'] = start_date
            df_user_workout['Weeks'] = num_weeks
            df_user_workout['Days Per Week'] = days_per_week
            df_user_workout['Cycles'] = num_cycles
            df_user_workout['Workout-Day-ID'] = df_user_workout.apply(lambda row: extract_workout_day_id(row), axis=1)

            df_workout_expanded = pd.concat(df_user_workout.apply(lambda row: generate_workout(row, df_workout_details), axis=1).tolist(), ignore_index=True)

            df_workout_expanded_complete = pd.concat(df_workout_expanded.apply(lambda row: generate_workout_details(row, df_excercises, df_user_specific_logs), axis=1).tolist(), ignore_index=True)

            st.subheader("Recommended Workout for Today")
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            df_workout_expanded_complete_filtered = df_workout_expanded_complete[df_workout_expanded_complete['Workout Date'] == today]
            st.dataframe(df_workout_expanded_complete_filtered[['Exercise', 'Weight', 'Units']])

            fig_exercise_timeline = px.scatter(df_workout_expanded_complete, x='Workout Date', y='Weight', color='Exercise').update_traces(mode='markers')
            st.plotly_chart(fig_exercise_timeline, use_container_width=True)

            with st.expander("Show all Workout Data"):
                st.dataframe(df_workout_expanded_complete)



def gather_data():
    gsheet_id = st.secrets["GSHEET_WORKOUT_ID"]
    gservice_data = st.secrets["GSERVICE_JSON_DATA"]

    # gservice_authorization = ast.literal_eval(gservice_data)
    gsclient = pygsheets.authorize(service_account_json=gservice_data)
    spreadsheet = gsclient.open_by_key(gsheet_id)

    # Read from each sheet in the spreadsheet
    worksheet = spreadsheet.worksheet_by_title('exercises')
    data = worksheet.get_all_records()
    df_excercises = pd.DataFrame(data)

    exercise_columns = df_excercises.columns
    base_columns = ['Exercise', 'CNJ Ratio', 'Units']
    exercise_columns = [item for item in exercise_columns if item not in base_columns]

    for c in exercise_columns:
        # Step 1: Replace empty strings or None with 'FALSE'
        df_excercises[c] = df_excercises[c].fillna('FALSE')

        # Step 2: Convert 'TRUE'/'FALSE' to boolean True/False
        df_excercises[c] = df_excercises[c].apply(lambda x: x == 'TRUE')

    users = spreadsheet.worksheet_by_title('users')
    user_data = users.get_all_records()
    df_users = pd.DataFrame(user_data)
    workouts = spreadsheet.worksheet_by_title('workouts')
    workouts_data = workouts.get_all_records()
    df_workouts = pd.DataFrame(workouts_data)
    workout_details = spreadsheet.worksheet_by_title('workout-details')
    workout_details_data = workout_details.get_all_records()
    df_workout_details = pd.DataFrame(workout_details_data)

    # Step 1: Replace empty strings or None with 'FALSE'
    df_workout_details['Random'] = df_workout_details['Random'].fillna('FALSE')

    # Step 2: Convert 'TRUE'/'FALSE' to boolean True/False
    df_workout_details['Random'] = df_workout_details['Random'].apply(lambda x: x == 'TRUE')

    user_logs = spreadsheet.worksheet_by_title('users-logs')
    user_logs_data = user_logs.get_all_records()
    df_user_logs = pd.DataFrame(user_logs_data)

    with st.expander('Excercises'):
        st.dataframe(df_excercises)
    with st.expander('Users', expanded=True):
        st.dataframe(df_users)
    with st.expander('User Logs'):
        st.dataframe(df_user_logs)
    with st.expander('Workouts'):
        st.dataframe(df_workouts)
    with st.expander('Workout Details'):
        st.dataframe(df_workout_details)

    return df_users, df_excercises, df_workouts, df_workout_details, df_user_logs


if __name__ == '__main__':
    app()
