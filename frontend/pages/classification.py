import requests
import os
import streamlit as st
from PIL import Image


API_BACKEND_URL = os.getenv("BACKEND_URL", "http://0.0.0.0:8000")

def main():
	st.title("Medical Classifier")
	st.write("Upload an image to classify")

	uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
	version = st.selectbox("Model's version", ["v1"])

	if uploaded_file is not None:
		image = Image.open(uploaded_file)
		st.image(image, caption="Uploaded Image", use_container_width=True)

		response = requests.post(API_BACKEND_URL, file={"file":uploaded_file,
												  		"model_version": version})
		if response.status_code == 200:
			result = response.json()
			st.success(f"Label: {result['label']} ({result['confidence'] * 100:.2f}%)")
		else:
			st.error("Failed to classify image")


if __name__ == "__main__":
	main()