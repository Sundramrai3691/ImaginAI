# ImaginAI

**ImaginAI** is a lightweight web application that uses neural style transfer to transform your images into AI-generated artwork. Built with Python and Streamlit, it enables you to upload content and style images to create high-quality stylized outputs.

## Features

- Upload your own content and style images
- Choose from a built-in gallery of styles
- Adjustable output resolution, style intensity, and training epochs
- Optional post-processing filters: sharpen, blur, enhance
- Download final output as HD JPEG or PNG

## Tech Stack

- Python
- Streamlit
- PyTorch
- Pillow (PIL)
- TensorFlow (for model architecture)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Sundramrai3691/ImaginAI.git
   cd ImaginAI

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the application:
   ```bash
   streamlit run app.py
