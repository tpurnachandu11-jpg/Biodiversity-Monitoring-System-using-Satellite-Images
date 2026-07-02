import streamlit as st
import ee
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import requests
import io
from datetime import datetime
from PIL import Image
from geopy.geocoders import Nominatim

# Configure Google Generative AI
GOOGLE_API_KEY = "AIzaSyDO4Jy1s_pTxg9y6qEFZNMfnPPYfmJ6A98"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Load pre-trained classification model
CLASSIFICATION_MODEL_PATH = "fine_tuned_habitat_classifier.h5"
classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)

# Class labels
CLASS_LABELS = [
    'River', 'AnnualCrop', 'SeaLake', 'Highway', 'Residential',
    'HerbaceousVegetation', 'PermanentCrop', 'Industrial', 'Forest', 'Pasture'
]

def initialize_earth_engine():
    """Authenticate and initialize Google Earth Engine."""
    try:
        ee.Authenticate()
        ee.Initialize(project="ee-vathsalyareddyg-hpred")
    except Exception as e:
        st.error(f"Error initializing Earth Engine: {e}")

def divide_into_tiles(image, tile_size, overlap=0):
    """Divide an image into smaller tiles."""
    img_width, img_height = image.size
    tiles = []
    step = tile_size - overlap
    for y in range(0, img_height - tile_size + 1, step):
        for x in range(0, img_width - tile_size + 1, step):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tiles.append(tile)
    return tiles

def reverse_geocode(latitude, longitude):
    geolocator = Nominatim(user_agent="geopy_example")
    location = geolocator.reverse((latitude, longitude))

    if location:
        address = location.address
        return address
    else:
        return None

def get_region_from_coordinates(lat, lon):

    result = reverse_geocode(lat, lon)

    if result:
        print(f"Address: {result}")
    else:
        print(f"Could not find address for the provided coordinates.")
    return result

def classify_tiles(tiles):
    """Classify image tiles using the trained model."""
    results = []
    for tile in tiles:
        tile_resized = tile.resize((224, 224)).convert("RGB")
        tile_array = np.expand_dims(np.array(tile_resized) / 255.0, axis=0)
        predictions = classification_model.predict(tile_array, verbose=0)
        predicted_class = np.argmax(predictions)
        results.append(CLASS_LABELS[predicted_class])
    return results

def fetch_sentinel_image(coords, date):
    """Fetch Sentinel-2 image for a given location and date."""
    region = ee.Geometry.Rectangle(coords)
    collection = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(region) \
        .filterDate(ee.Date(date), ee.Date(date).advance(1, 'day')) \
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    return collection.first()

def analyze_biodiversity(region, classifications):
    """Analyze biodiversity and predict future habitat conditions."""
    prompt = (
        f"Analyze biodiversity in {region}. Detected classifications: {classifications}. "
        "Predict future habitat conditions based on environmental changes, climate patterns, "
        "and human impact. Suggest conservation strategies."
    )
    response = model.generate_content(prompt)
    return response.text

def reverse_geocode(lat, lon):
    """Retrieve address from latitude and longitude."""
    geolocator = Nominatim(user_agent="geo_analyzer")
    location = geolocator.reverse((lat, lon))
    return location.address if location else None

def analyze_image_differences(image1, image2):
    """Analyze differences between two images."""
    diff_array = np.abs(np.array(image1.convert("L")) - np.array(image2.convert("L")))
    return Image.fromarray(diff_array), np.sum(diff_array)

def detect_ndvi_anomalies(image1, image2, threshold=50):
    """Detect NDVI anomalies between two images."""
    diff_array = np.array(image2.convert("L")) - np.array(image1.convert("L"))
    anomaly_mask = np.abs(diff_array) > threshold
    anomaly_image = np.zeros_like(diff_array)
    anomaly_image[anomaly_mask] = 255
    return Image.fromarray(anomaly_image), np.sum(anomaly_mask)

def generate_insights(diff_scores):
    """Generate AI-based insights based on image differences."""
    prompt = (
        f"Analyze habitat changes using these image differences: {diff_scores}. "
        "Prioritize insights from image differences, focusing on biodiversity trends "
        "and possible conservation measures."
    )
    response = model.generate_content(prompt)
    return response.text

def generate_anomaly_insights(anomaly_scores, time_ranges, location):
    """Generate insights based on detected NDVI anomalies."""
    prompt = (
        f"Detected NDVI anomalies in {location} during {time_ranges}: {anomaly_scores}. "
        "Analyze the potential causes (environmental, climatic, human-induced factors). "
        "Suggest conservation strategies and predict futire habitat conditions based on give {location} fin out the biome and back your prediction with some quoted facts like news articles"
    )
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("Sentinel-2 Image Processing and Classification")

    option = st.radio("Choose Analysis Type:", ["Single Image Classification", "Temporal Analysis & Anomaly Detection"])

    if option == "Single Image Classification":
        uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            tiles = divide_into_tiles(image, 64)
            classifications = classify_tiles(tiles)
            st.write("### Classification Results", classifications)

    elif option == "Temporal Analysis & Anomaly Detection":
        min_lon, min_lat = st.number_input("Min Longitude", value=-55.5), st.number_input("Min Latitude", value=-7.2)
        max_lon, max_lat = st.number_input("Max Longitude", value=-55.3), st.number_input("Max Latitude", value=-7.5)
        date1_start = st.date_input("Select First Date Range Start")
        date1_end = st.date_input("Select First Date Range End")
        date2_start = st.date_input("Select Second Date Range Start")
        date2_end = st.date_input("Select Second Date Range End")

        region = get_region_from_coordinates((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
        st.write(f"**Detected Region:** {region}")

        if st.button("Fetch, Process, and Compare Images"):
            try:
                roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

                # Time ranges selection
                time_ranges = [
                    (str(date1_start), str(date1_end)),
                    (str(date2_start), str(date2_end))
                ]
                
                all_images = []
                diff_scores = []
                anomaly_scores = []

                # Fetch images for both time ranges
                for start_date, end_date in time_ranges:
                    try:
                        dataset = ee.ImageCollection("COPERNICUS/S2") \
                            .filterBounds(roi) \
                            .filterDate(start_date, end_date) \
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))

                        rgb_bands = dataset.select(['B4', 'B3', 'B2'])
                        median_image = rgb_bands.median().clip(roi)

                        vis_params = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
                        thumb_url = median_image.getThumbURL({
                            'region': roi, 'dimensions': [512, 512], 'format': 'PNG', **vis_params
                        })

                        response = requests.get(thumb_url)
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")

                        st.image(image, caption=f"Image ({start_date} to {end_date})", use_container_width=True)
                        all_images.append(image)

                    except Exception as e:
                        st.error(f"Error processing time range {start_date} - {end_date}: {e}")

                # Analyze image differences
                if len(all_images) == 2:
                    diff_image, diff_score = analyze_image_differences(all_images[0], all_images[1])
                    anomaly_image, anomaly_score = detect_ndvi_anomalies(all_images[0], all_images[1])

                    st.image(diff_image, caption="Difference Between Time Ranges", use_container_width=True)
                    st.image(anomaly_image, caption="NDVI Anomaly Map", use_container_width=True)
                    st.write(f"Anomaly Score: {anomaly_score}")

                    diff_scores.append(diff_score)
                    anomaly_scores.append(anomaly_score)

                    # Generate AI insights
                    insights = generate_anomaly_insights(
                        diff_scores, f"{time_ranges[0]} vs {time_ranges[1]}",region
                    )

                    st.write("### AI-Based Interpretation:")
                    st.write(insights)

                else:
                    st.error("Error: Could not fetch images for both time ranges.")

                st.success("Processing completed successfully!")

            except Exception as e:
                st.error(f"Error processing images: {e}")


if __name__ == "__main__":
    initialize_earth_engine()
    main()