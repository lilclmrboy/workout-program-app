import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import logging
import pygsheets
import ast
import hmac
import hashlib

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
log = logging.getLogger(__name__)
logging.basicConfig(format=FORMAT)
log.setLevel(logging.DEBUG)

if 'username_hash' not in st.session_state:
    st.session_state.username_hash = None


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
        """Form with widgets to collect user information"""
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
            st.write(f"Username hash: {username_hash}")
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


if not check_password():
    st.stop()

st.title("🎈 Workout Planner")
st.write(
    "Workout generation Utility"
)

gsheet_id = st.secrets["GSHEET_WORKOUT_ID"]
gservice_data = st.secrets["GSERVICE_JSON_DATA"]

gservice_authorization = ast.literal_eval(gservice_data)
gsclient = pygsheets.authorize(service_account_json=gservice_data)
spreadsheet = gsclient.open_by_key(gsheet_id)

worksheet = spreadsheet.worksheet_by_title('exercises')
data = worksheet.get_all_records()
df_excercises = pd.DataFrame(data)

users = spreadsheet.worksheet_by_title('users')
user_data = users.get_all_records()
df_users = pd.DataFrame(user_data)

with st.expander('Excercises'):
    st.dataframe(df_excercises)

with st.expander('Users', expanded=True):
    st.dataframe(df_users)

flag_submitted = False
with st.form('Configuration'):
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

    flag_submitted = st.form_submit_button("Generate")

if flag_submitted:
    st.write(f'Generating workout for {st.session_state.username_hash}')

    users = df_users['username_hash'].unique()
    if st.session_state.username_hash in users:
        st.write('Success! User has an account!')

    # Generate the range of dates
    dates = pd.date_range(start=start_date, periods=num_weeks * 3, freq='B').to_series()

    # Filter to only include Mondays, Wednesdays, and Fridays
    dates = dates[dates.dt.dayofweek.isin([0, 2, 4])]

    st.write(dates)


