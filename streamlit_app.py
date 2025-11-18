
import streamlit as st
import numpy as np
import pickle
import json
import pandas as pd

# Initialize variables before attempting to load
modelos_por_municipio = {}
municipios_features = {}

# Load the trained models and data (only once at the start of the Streamlit app)
try:
    with open("modelos_por_municipio.pkl", "rb") as f:
        modelos_por_municipio = pickle.load(f)
    with open("municipios_features.json", "r") as f:
        municipios_features = json.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'modelos_por_municipio.pkl' and 'municipios_features.json' are saved after training.")
    # The variables are already initialized, so no need to stop here,
    # but the app will show the warning about no models available.

# Assuming the features for each municipality are the same after preprocessing
# You might need to adjust this if features vary significantly between municipalities
# Get the list of features from one of the municipalities (e.g., the first one)
if modelos_por_municipio:
    first_municipio = list(modelos_por_municipio.keys())[0]
    all_features = municipios_features.get(first_municipio, [])
    # Exclude the target variable and any features not used for prediction
    features_for_prediction = [f for f in all_features if f not in ['precio', 'muni', 'col', 'cord', 'price_per_sq_meter', 'distance_from_center']]
else:
    st.warning("No trained models found.")
    features_for_prediction = []


def predict_price(municipio, data, modelos, municipios_features):
    """
    Predicts the price of a property based on the municipality and input data.

    Args:
        municipio (str): The selected municipality.
        data (dict): A dictionary containing the input features.
        modelos (dict): A dictionary of trained models for each municipality.
        municipios_features (dict): A dictionary containing the feature names for each municipality.

    Returns:
        float: The predicted price.
    """
    if municipio in modelos and municipio in municipios_features:
        model = modelos[municipio]
        features = municipios_features[municipio]

        # Create a DataFrame from input data to match model's expected input
        input_df = pd.DataFrame([data])

        # Ensure the order of columns matches the training data
        # Add any missing columns with default values (e.g., 0) if your model expects them
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[features]


        prediction = model.predict(input_df)[0]
        return prediction
    else:
        return None


def main():
    # --- Add an image and title side-by-side ---
    col1, col2 = st.columns([1, 3], vertical_alignment="center") # Adjust ratios as needed
    with col1:
        st.image("/content/imagen.jpeg", caption="Mi Imagen", width=175)
    with col2:
        st.title("House Price Prediction by Municipality")
    # -------------------------------------------

    st.write("Select a municipality and enter property details to get an estimated price.")

    # Get the list of municipalities with trained models
    municipios_list = list(modelos_por_municipio.keys())

    if not municipios_list:
        st.warning("No models available for prediction. Please train the models first.")
        return

    # Select municipality
    chosen_municipio = st.selectbox("Select Municipality:", municipios_list)

    # Get features for the chosen municipality
    if chosen_municipio in municipios_features:
        features = municipios_features[chosen_municipio]
    else:
        st.warning(f"Features not found for {chosen_municipio}.")
        return


    # Input fields for features (adjust based on your model's features)
    # Exclude 'precio' and municipality specific features if they were used
    input_data = {}
    for feature in features:
        if feature not in ['precio', 'muni', 'col', 'cord', 'price_per_sq_meter', 'distance_from_center']: # Exclude target and engineered features
             # You'll need to add appropriate input widgets for each feature type
            if feature in ['const', 'cuart', 'sanit', 'piso', 'terr']: # Example: numeric inputs
                 input_data[feature] = st.number_input(f"Enter {feature}:", value=0.0)
            elif feature in ['lat', 'long']: # Example: numeric inputs
                 input_data[feature] = st.number_input(f"Enter {feature}:", value=0.0, format="%.6f")
            # Add more conditions for other data types if needed

    # Add input for engineered features if they are needed as input (e.g. if they are not calculated in the predict function)!
    # For now, we assume price_per_sq_meter and distance_from_center are calculated internally if needed by the model

    # Define columns for the button and prediction output
    col_btn, col_prediction_output = st.columns([1, 3]) # Adjust ratios as needed

    with col_btn:
        estimate_button_clicked = st.button('Estimate Price')

    if estimate_button_clicked:
        # You might need to calculate engineered features here based on user input
        # For example:
        if 'const' in input_data and input_data['const'] > 0:
             input_data['price_per_sq_meter'] = None # Or calculate if needed from input
        else:
             input_data['price_per_sq_meter'] = None

        # You might need the mean lat/long for the chosen municipality to calculate distance_from_center
        # This would require saving mean_lat and mean_long per municipality or recalculating
        input_data['distance_from_center'] = None # Or calculate if needed from input

        predicted_price = predict_price(chosen_municipio, input_data, modelos_por_municipio, municipios_features)

        if predicted_price is not None:
            st.session_state['last_predicted_price'] = predicted_price
        else:
            st.session_state['last_predicted_price'] = None

    # Display the last predicted price and range in the second column if available
    if 'last_predicted_price' in st.session_state and st.session_state['last_predicted_price'] is not None:
        predicted_price_to_display = st.session_state['last_predicted_price']
        with col_prediction_output: # Display in the second column
            st.success(f"El precio estimado es: {predicted_price_to_display:,.2f}")
            lower_bound = predicted_price_to_display * 0.90 # 10% total range, so 5% lower
            upper_bound = predicted_price_to_display * 1.10 # 5% higher
            st.markdown(f"**Rango de Precio (±10%):** {lower_bound:,.2f} - {upper_bound:,.2f}")
    elif 'last_predicted_price' in st.session_state and st.session_state['last_predicted_price'] is None:
        with col_prediction_output:
            st.error("No se pudo predecir el precio. Por favor, revisa tus entradas y el municipio seleccionado.")


main()
