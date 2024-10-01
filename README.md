Hand Sign Detector Using CNN

This project implements a Convolutional Neural Network (CNN) model using PyTorch to detect and identify hand signs in real-time. The system utilizes OpenCV to access the camera, capturing live video feeds and processing the frames to detect specific hand signs. The model has been trained on a dataset of labeled hand sign images, ensuring accurate recognition.

Features

	•	Real-time Hand Sign Detection: Uses your device’s camera to detect and recognize hand signs.
	•	Trained CNN Model: The CNN is trained on a robust dataset of hand sign images to provide accurate classification.
	•	Assistive Communication Tool: Designed to assist individuals in communicating through hand signs, making it a valuable resource for non-verbal communication.
	•	Scalable and Extendable: The model can be further trained or fine-tuned to detect additional hand signs or gestures.

Technology Stack

	•	Framework: PyTorch for building and training the CNN model.
	•	Computer Vision: OpenCV for capturing live video and processing image frames.
	•	Python: Main programming language used for the project.

How It Works

	1.	Camera Input: OpenCV captures real-time video feed from your camera.
	2.	Preprocessing: The captured frames are preprocessed and passed through the CNN for classification.
	3.	Prediction: The CNN model outputs the predicted hand sign based on the trained dataset.
	4.	Output: The detected hand sign is displayed on the screen, providing real-time feedback.
