import pandas as pd
import random
import streamlit as st
import streamlit_authenticator as stauth
from streamlit import session_state as ss
import logging
import pygsheets
import ast
import hmac
import hashlib
import time
import math
import plotly.express as px
from datetime import datetime
from scipy import signal
import numpy as np

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
log = logging.getLogger(__name__)
logging.basicConfig(format=FORMAT)
log.setLevel(logging.DEBUG)

if 'username_hash' not in st.session_state:
    st.session_state.username_hash = None

if 'df_users' not in st.session_state:
    st.session_state.df_users = pd.DataFrame()

if 'df_excercises' not in st.session_state:
    st.session_state.df_excercises = pd.DataFrame()

if 'df_workouts' not in st.session_state:
    st.session_state.df_workouts = pd.DataFrame()

if 'df_workout_details' not in st.session_state:
    st.session_state.df_workout_details = pd.DataFrame()

if 'df_user_logs' not in st.session_state:
    st.session_state.df_user_logs = pd.DataFrame()

if 'df_today' not in st.session_state:
    st.session_state.df_today = pd.DataFrame()

if 'ed' not in st.session_state:
    st.session_state.ed = {}

if 'flag_first_run' not in st.session_state:
    st.session_state.flag_first_run = True


def periodization_equation(step, nTotalSteps, base=0.6, nCycles=3.0):
    time_steps_ss = np.linspace(start=1, stop=1, num=nTotalSteps, endpoint=False)
    time_steps = np.linspace(start=0, stop=1, num=nTotalSteps, endpoint=False)
    vals = signal.sawtooth(2 * np.pi * nCycles * time_steps, width=1.0) + time_steps

    vals_min = np.min(vals)
    vals = vals - vals_min
    vals_max = np.max(vals)
    vals = vals / vals_max

    growth_delta = 1.0 - base
    vals = vals * growth_delta
    base_vals = time_steps_ss * base
    percent = base_vals[step] + vals[step]
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
    try:
        workout_day_id = index_val % days_per_week
    except ValueError as e:
        workout_day_id = None
    return workout_day_id


def generate_periodization(row, workout_id, num_workouts):
    cycles = row.get('Cycles')
    workout_step = row.get('ID')

    periodization_val = periodization_equation(step=workout_step,
                                               nTotalSteps=num_workouts,
                                               base=0.6,
                                               nCycles=cycles)
    return periodization_val


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
    weeks = row.get('Weeks')
    days_per_week = row.get('Days Per Week')
    workout_percentage = row.get('Workout Percentage')
    time_val = row.get('Time')
    improvement_percentage = row.get('Improvement Percentage')

    _df_excercise = df_excercises[df_excercises['Exercise'] == exercise]
    if flag_random == True:
        _df_excercise = df_excercises[df_excercises[exercise_type] == True].sample(n=1)

    excercise_val = _df_excercise.iloc[0]['Exercise']
    units_val = _df_excercise.iloc[0]['Units']

    try:

        df_filtered = df_user_logs[df_user_logs['Exercise'] == excercise_val]

        if df_filtered.empty:
            df_filtered = df_user_logs[df_user_logs['Exercise'] == 'BodyWeight']
            df_filtered['date'] = pd.to_datetime(df_filtered['date'], format='%m/%d/%Y')
            df_sorted = df_filtered.sort_values(by='date', ascending=False)
            target_weight = df_sorted.iloc[0]['Weight']
            ratio_val = 1.0

    except KeyError as e:
        logging.debug(f"Error getting bodyweight: {e}")
        target_weight = 100.0
        ratio_val = _df_excercise.iloc[0]['CNJ Ratio']

    exercise_defintions = []
    for s in range(sets):

        percent_step_size = 0.1
        percent_calc = 1.0 - (percent_step_size * sets) + (percent_step_size * (s + 1))

        if progression == 'Linear':

            if exercise_type in ['Core', 'Bodyweight']:
                repetitions_val = int(percent_calc * row.get('Repetitions', repititions)) * improvement_percentage
                weight_val = 0.0
            else:
                weight_val = target_weight * percent_calc * ratio_val * improvement_percentage
                repetitions_val = row.get('Repetitions', repititions)

        else:  # Applies for both 'Flat' and other cases
            weight_val = target_weight * workout_percentage * ratio_val
            repetitions_val = repititions

        # Adjust weight and repetitions based on exercise type
        if exercise_type not in ['Core', 'Bodyweight']:
            weight_val = max(round(weight_val / 2.5) * 2.5, 40.0)
            repetitions_val = adjust_repetitions(workout_percentage, repetitions_val)
        else:
            weight_val = 0
        weight_val = workout_percentage * weight_val

        _exercise = {
            'Workout-ID': workout_id,
            'Workout Percentage': workout_percentage,
            'Workout-Day-ID': workout_day_id,
            'Workout Date': workout_date,
            'Workout Start Date': start_date,
            'Exercise-Set-ID': s,
            'Exercise-Type': exercise_type,
            'Exercise': excercise_val,
            'CNJ Raio': ratio_val,
            'Repetitions': repetitions_val,
            'Weight': weight_val,
            'Time': time_val,
            'Units': units_val,
            'Random Flag': flag_random,
        }

        exercise_defintions.append(_exercise)

    return pd.DataFrame(exercise_defintions)


def adjust_repetitions(workout_percentage, repetitions_val):
    adjustments = [
        (0.975, 0.25),
        (0.95, 0.333),
        (0.925, 0.5),
        (0.9, 0.75)
    ]

    for threshold, factor in adjustments:
        if workout_percentage >= threshold:
            repetitions_val = int(repetitions_val * factor)
            break

    repetitions_val = max(1, repetitions_val)
    return repetitions_val


def data_editor_changed(df):
    try:
        _df = ss.df_today  # Make a copy of the DataFrame
        edited_rows = st.session_state.ed["edited_rows"]  # Ensure correct access to session state
        logging.debug("edited_rows: ", edited_rows)  # Debugging log

        index_val_offset = int(df.index[0])

        rows_to_keep = []
        for key in edited_rows.keys():
            rows_to_keep.append(int(key) + index_val_offset)

        # Apply the edits to the DataFrame
        for row_index, row_data in edited_rows.items():
            for col_name, new_value in row_data.items():
                _df.at[int(row_index + index_val_offset), col_name] = new_value  # Update the DataFrame with new values
                _df.at[int(row_index + index_val_offset), 'Workout Date'] = pd.Timestamp.now()

        ss.df_today = _df

        __df = _df.loc[rows_to_keep]
        log_updates = __df.values.tolist()
        update_users_logs(log_updates, list(__df.columns))

    except AttributeError as e:
        logging.debug(f"Error showing changes to dataframe: {e}")
        st.write(f"Error: {e}")


def update_users_logs(log_updates, headers):
    gsheet_id = st.secrets["GSHEET_WORKOUT_ID"]
    gservice_data = st.secrets["GSERVICE_JSON_DATA"]

    gsclient = pygsheets.authorize(service_account_json=gservice_data)
    spreadsheet = gsclient.open_by_key(gsheet_id)

    # Read from each sheet in the spreadsheet
    worksheet = spreadsheet.worksheet_by_title('users-logs')

    # Get the last row with data
    def get_last_row(sheet):
        # Use A:A to get all values in column A
        values = sheet.get_col(1, include_tailing_empty=False)
        return len(values)

    last_row = get_last_row(worksheet)

    # Calculate the starting cell for appending
    start_cell = f'A{last_row + 1}'

    converted_data_iso = []

    if start_cell == 'A1':
        converted_data_iso = [headers]

    for log_update in log_updates:
        _converted_data_iso = [
            item.isoformat() if isinstance(item, pd.Timestamp) else item
            for item in log_update
        ]
        converted_data_iso.append(_converted_data_iso)

    worksheet.update_values(start_cell, converted_data_iso)


def app():
    if not check_password():
        st.stop()

    st.title("🎈 Workout Planner")

    if st.sidebar.button("Refresh Data"):
        with st.spinner('Collecting Workout Data'):
            st.session_state.df_users, st.session_state.df_excercises, st.session_state.df_workouts, st.session_state.df_workout_details, st.session_state.df_user_logs = gather_data()
            ss.df_today = pd.DataFrame()

    if ss.flag_first_run:
        with st.spinner('Collecting Workout Data'):
            st.session_state.df_users, st.session_state.df_excercises, st.session_state.df_workouts, st.session_state.df_workout_details, st.session_state.df_user_logs = gather_data()
            ss.df_today = pd.DataFrame()
            ss.flag_first_run = False

    if not st.session_state.df_users.empty:

        df_users = st.session_state.df_users
        df_excercises = st.session_state.df_excercises
        df_workouts = st.session_state.df_workouts
        df_workout_details = st.session_state.df_workout_details
        df_user_logs = st.session_state.df_user_logs

        users = df_users['username_hash'].unique()
        if st.session_state.username_hash in users:

            _username = st.session_state.username_hash

            df_user_workouts = df_users.merge(df_workouts, on=['Workout-ID'], how='left')
            df_user_workouts = df_user_workouts[df_user_workouts['username_hash'] == _username]
            df_user_workouts['DescriptionTime'] = df_user_workouts.apply(lambda row: expand_description(row), axis=1)
            df_user_workouts['Start Date'] = pd.to_datetime(df_user_workouts['Start Date'])
            try:
                df_user_specific_logs = df_user_logs[df_user_logs['username_hash'] == _username]
            except KeyError as e:
                df_user_specific_logs = pd.DataFrame()

            # Select the workout from the user
            workout_choice = st.sidebar.selectbox('Workout Selection',
                                                  options=df_user_workouts['DescriptionTime'].unique())

            df = df_user_workouts[df_user_workouts['DescriptionTime'] == workout_choice]
            workout_ids = df['Workout-ID'].unique()
            for workout_id in workout_ids:

                df_user_workout = create_user_workout_meta_information(df, workout_id)
                df_workout_expanded_complete = create_user_workout_program_information(df_user_workout,
                                                                                       df_workout_details,
                                                                                       df_excercises,
                                                                                       df_user_specific_logs)

                st.subheader("Recommended Workout for Today")
                create_workout_for_today(_username, df_workout_expanded_complete)

                with st.expander("Show all Workouts", expanded=True):

                    df_workout_expanded_complete = df_workout_expanded_complete.merge(
                        df_workouts[['Workout-ID', 'Description', 'Workout Weeks']], on=['Workout-ID'])

                    workout_dates = df_workout_expanded_complete['Workout Date'].unique()
                    for workout_date in workout_dates:
                        display_workout_cycle_unique_date(df_workout_expanded_complete, df_workouts, workout_date)


def display_workout_cycle_unique_date(df_workout_expanded_complete, df_workouts, workout_date):
    date_str = workout_date.strftime("%A - %B %-d, %Y")
    st.header(f"{date_str}")
    _df = df_workout_expanded_complete[df_workout_expanded_complete['Workout Date'] == workout_date]
    workouts = _df['Workout-ID'].unique()
    for workout in workouts:
        __df = _df[_df['Workout-ID'] == workout]

        workout_str = df_workouts.iloc[0]['Description']

        percent_effort = __df.iloc[0]['Workout Percentage'] * 100.0
        st.subheader(f"{workout_str} at {percent_effort:.0f} percent")

        _df_exercises = __df.groupby(
            ['Workout-Day-ID', 'Exercise', 'Repetitions', 'Weight', 'Weight (lbs)',
             'Time']).size().to_frame(name='Sets').reset_index()
        _df_exercises_summary = _df_exercises[
            ['Workout-Day-ID', 'Exercise', 'Sets', 'Repetitions', 'Weight', 'Weight (lbs)', 'Time']]

        _df_exercises_summary.rename(columns={'Repetitions': "Reps"}, inplace=True)
        df_styled = _df_exercises_summary.style.set_properties(
            subset=['Weight', 'Weight (lbs)', 'Reps', 'Sets', 'Time'], **{'text-align': 'center'})
        df_styled = df_styled.format(precision=1).hide(axis="index")
        st.markdown(df_styled.to_html(), unsafe_allow_html=True)


def create_user_workout_program_information(df_user_workout, df_workout_details, df_excercises, df_user_specific_logs):
    df_workout_expanded = pd.concat(
        df_user_workout.apply(lambda row: generate_workout(row, df_workout_details), axis=1).tolist(),
        ignore_index=True)
    df_workout_expanded_complete = pd.concat(df_workout_expanded.apply(
        lambda row: generate_workout_details(row, df_excercises, df_user_specific_logs), axis=1).tolist(),
                                             ignore_index=True)
    df_workout_expanded_complete['Weight (lbs)'] = df_workout_expanded_complete['Weight'].apply(
        lambda x: round_weight(x, units='lbs'))
    df_workout_expanded_complete['Weight'] = df_workout_expanded_complete['Weight'].apply(
        lambda x: round_weight(x, units='kg'))
    return df_workout_expanded_complete


def create_user_workout_meta_information(df, workout_id):
    start_date = df.loc[df['Workout-ID'] == workout_id, 'Start Date'].iloc[0]
    days_of_week = df.loc[df['Workout-ID'] == workout_id, 'Days of Week'].iloc[0]
    days_of_week = ast.literal_eval(days_of_week)
    num_weeks = df.loc[df['Workout-ID'] == workout_id, 'Workout Weeks'].iloc[0]
    days_per_week = df.loc[df['Workout-ID'] == workout_id, 'DaysPerWeek'].iloc[0]
    num_cycles = df.loc[df['Workout-ID'] == workout_id, 'Cycles'].iloc[0]
    improvement_percentage = df.loc[df['Workout-ID'] == workout_id, 'Improvement Percentage'].iloc[0]
    # Generate the range of dates
    dates = pd.date_range(start=start_date, periods=num_weeks * 7, freq='D').to_series()
    # Filter to only include the days we care about.
    dates = dates[dates.dt.dayofweek.isin(days_of_week)]
    df_user_workout = pd.DataFrame(dates)
    df_user_workout.reset_index(names='Workout Date', inplace=True)
    df_user_workout.drop(columns=0, inplace=True)
    df_user_workout['Workout-ID'] = workout_id
    df_user_workout['Start Date'] = start_date
    df_user_workout['Weeks'] = num_weeks
    df_user_workout['Days Per Week'] = days_per_week
    df_user_workout['Cycles'] = num_cycles
    df_user_workout['Improvement Percentage'] = 1.0 + improvement_percentage / 100.0
    try:
        df_user_workout['Workout-Day-ID'] = df_user_workout.apply(lambda row: extract_workout_day_id(row),
                                                                  axis=1)
    except ValueError as e:
        logging.debug(f"Error setting workout day id: {e}")
    df_user_workout = df_user_workout.reset_index().rename(columns={'index': 'ID'})
    df_user_workout['Workout Percentage'] = df_user_workout.apply(
        lambda row: generate_periodization(row, row.index, len(df_user_workout)), axis=1)
    return df_user_workout


def create_workout_for_today(_username, df_workout_expanded_complete):

    if ss.df_today.empty:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ss.df_today = df_workout_expanded_complete[df_workout_expanded_complete['Workout Date'] == today]

        ss.df_today['Weight (lbs)'] = ss.df_today['Weight'].apply(lambda x: round_weight(x, units='lbs'))
        ss.df_today['Weight'] = ss.df_today['Weight'].apply(lambda x: round_weight(x, units='kg'))

        ss.df_today['Completed'] = False
        ss.df_today['Missed'] = False
        ss.df_today['User'] = _username

    _workout_fields = ['Exercise', 'Repetitions', 'Weight (lbs)', 'Weight', 'Completed', 'Missed']
    _df_workout_today_todo = ss.df_today[(ss.df_today['Completed'] == False)]

    st.subheader('Next Exercise')

    df_next_exercise = _df_workout_today_todo.head(1)

    # Replace empty strings with NaN
    df_next_exercise['Time'] = df_next_exercise['Time'].replace('', np.nan)

    # Convert to numeric (this will turn non-convertible values into NaN)
    df_next_exercise['Time'] = pd.to_numeric(df_next_exercise['Time'], errors='coerce')

    # Replace NaN values with 0.0
    df_next_exercise['Time'] = df_next_exercise['Time'].fillna(0.0)

    st.data_editor(df_next_exercise,
                   column_config={
                       'TODO': st.column_config.CheckboxColumn(
                           label="Completed",
                           help="Check if the exercise is completed",
                       ),
                       'Weight': st.column_config.NumberColumn(
                           help='Weight of exercise in SI units',
                           label='Weight (kg)',
                           format="%.1f",
                       ),
                       'Weight (lbs)': st.column_config.NumberColumn(
                           help='Weight of exercise in Imperial units',
                           format="%.0f",
                       ),
                   },
                   hide_index=True,
                   on_change=data_editor_changed,
                   args=[df_next_exercise],
                   column_order=_workout_fields,
                   disabled=['Exercise', 'Repetitions', 'Weight (lbs)', 'Weight', 'Time'],
                   use_container_width=True,
                   key="ed",
                   )

    set_time = df_next_exercise['Time'].iloc[0]
    if set_time > 0.0 and st.button(f"Begin Timer for {set_time} seconds"):
        timer_bar = st.progress(0, text=f"{set_time} seconds remaining")
        for percent_complete in range(100):
            time_delta = set_time / 100.0
            time.sleep(time_delta)
            time_remaining = set_time - time_delta * percent_complete
            progress_text = f"{int(time_remaining)} seconds remaining"
            timer_bar.progress(percent_complete + 1, text=progress_text)

    st.subheader('Upcoming Exercises')
    if _df_workout_today_todo.empty:
        st.write('All exercises have been completed.')
    else:
        _df_todo = _df_workout_today_todo.drop(columns=['Completed', 'Missed'])
        st.dataframe(_df_todo.tail(-1),
                     column_config={
                         'Weight': st.column_config.NumberColumn(
                             help='Weight of exercise in SI units',
                             label='Weight (kg)',
                             format="%.1f",
                         ),
                         'Weight (lbs)': st.column_config.NumberColumn(
                             help='Weight of exercise in Imperial units',
                             format="%.0f",
                         ),
                     },
                     use_container_width=True,
                     column_order=_workout_fields,
                     hide_index=True,
                     height=(len(_df_todo.tail(-1)) + 1) * 35 + 3,
                     )

    st.subheader('Completed Exercises')
    _df_completed = ss.df_today[ss.df_today['Completed'] == True].drop(columns=['Completed'])
    if _df_completed.empty:
        st.write('No exercises have been done today.')
    else:
        st.dataframe(_df_completed,
                     column_config={
                         'Weight': st.column_config.NumberColumn(
                             help='Weight of exercise in SI units',
                             label='Weight (kg)',
                             format="%.0f",
                         ),
                         'Weight (lbs)': st.column_config.NumberColumn(
                             help='Weight of exercise in Imperial units',
                             format="%.0f",
                         ),
                     },
                     use_container_width=True,
                     column_order=_workout_fields,
                     hide_index=True,
                     height=(len(_df_completed.tail(-1)) + 1) * 35 + 3,
                     )

    fig_exercise_timeline = px.scatter(ss.df_today, x='Exercise', y='Weight',
                                       color='Exercise').update_traces(mode='lines+markers')
    st.plotly_chart(fig_exercise_timeline, use_container_width=True)


def round_weight(x, units='kg'):
    if units == 'lbs':
        divisor = 5.0
        conversion = 2.204
        min_val = 45
    else:
        divisor = 2.5
        conversion = 1.0
        min_val = 20
    multiplier = (x * conversion) / divisor
    rounded_val = round(multiplier) * divisor
    return max(min_val, rounded_val)


def gather_data():
    gsheet_id = st.secrets["GSHEET_WORKOUT_ID"]
    gservice_data = st.secrets["GSERVICE_JSON_DATA"]

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

    # with st.expander('Excercises'):
    #     st.dataframe(df_excercises)
    # with st.expander('Users', expanded=False):
    #     st.dataframe(df_users)
    # with st.expander('User Logs'):
    #     st.dataframe(df_user_logs)
    # with st.expander('Workouts'):
    #     st.dataframe(df_workouts)
    # with st.expander('Workout Details'):
    #     st.dataframe(df_workout_details)

    return df_users, df_excercises, df_workouts, df_workout_details, df_user_logs


if __name__ == '__main__':
    app()
