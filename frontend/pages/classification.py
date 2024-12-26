import requests
import os
import streamlit as st


API_BACKEND_URL = os.getenv("BACKEND_URL", "http://0.0.0.0:8000")

def main():
	st.title("Medical Classifier")
	st.write("Upload an image to classify")

	uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

	if uploaded_file is not None:
		st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

		response = requests.post(API_BACKEND_URL, file={"file":uploaded_file})
		if response.status_code == 200:
			result = response.json()
			st.success(f"Label: {result['label']} ({result['confidence'] * 100:.2f}%)")
		else:
			st.error("Failed to classify image")


if __name__ == "__main__":
	main()