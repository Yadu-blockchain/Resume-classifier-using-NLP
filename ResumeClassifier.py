# Import necessary libraries
import json
import fitz  # PyMuPDF - used to extract text from PDF files
import docx  # To handle .docx Word documents
import os
import pytesseract  # OCR tool for extracting text from images
from PIL import Image  # Python Imaging Library for image processing

# Set the path to the Tesseract executable (required for Windows) in your PC
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract text from a JSON file
def extract_text_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if isinstance(data, dict):
            return ' '.join(str(value) for value in data.values())
        elif isinstance(data, list):
            return ' '.join(str(item) for item in data)
        return str(data)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to extract text from a Word (.docx) file
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Function to extract text from a PNG image using OCR
def extract_text_from_png(png_path):
    img = Image.open(png_path)
    text = pytesseract.image_to_string(img)
    return text

# Unified function to detect the file type and call the appropriate extractor
def load_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.json':
        return extract_text_from_json(file_path)
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.png':
        return extract_text_from_png(file_path)
    else:
        raise ValueError("Unsupported file format: " + ext)

# Logging prediction results to a CSV file
import csv
from datetime import datetime

def log_prediction(file_path, predicted_category, log_file='resume_logs.csv'):
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write headers if file doesn't exist
            writer.writerow(['Resume Number', 'File Path', 'Predicted Category', 'Timestamp'])
        
        # Get total resume count (excluding header)
        resume_number = sum(1 for line in open(log_file, encoding='utf-8')) if file_exists else 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([resume_number, file_path, predicted_category, timestamp])

# Load the resume dataset
import pandas as pd
# Provide the path where master data is saved in your personal computer
df = pd.read_csv("C:\\Users\\user\\Desktop\\Jupyter\\Projects\\Resume dataset\\UpdatedResumeDataSet.csv")

# Function to clean raw resume text
import re
def cleanResume(txt):
    cleanText = re.sub(r'http\S+', ' ', txt)                         
    cleanText = re.sub(r'RT|cc', ' ', cleanText)                     
    cleanText = re.sub(r'#\S+', ' ', cleanText)                      
    cleanText = re.sub(r'@\S+', ' ', cleanText)                      
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)             
    cleanText = re.sub(r'\s+', ' ', cleanText)                      
    return cleanText.strip()
    
# Apply cleaning function to all resumes in the dataset
df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# Encode the job categories to numerical format
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])

# Display encoded category IDs
df['Category'].unique()

# Convert text data into TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])

requiredtext = tfidf.transform(df['Resume'])

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(requiredtext, df['Category'], test_size=0.2, random_state=42)

# Path to the user's resume
# Provide the path where the resume data file is saved in your personal computer
resume_path = "C:\\Users\\user\\Desktop\\Jupyter\\Projects\\Resume dataset\\resume sample2.png"
my_resume = load_resume(resume_path)

# Build the classifier model using KNN (wrapped inside One-vs-Rest)
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Clean and transform user resume for prediction
cleaned_resume = cleanResume(my_resume)
input_features = tfidf.transform([cleaned_resume])
prediction_id = clf.predict(input_features)[0]

# Map prediction ID to the original category name
category_mapping = {
    0: "Advocate", 1: "Arts", 2: "Automation testing", 3: "Blockchain Developer",
    4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science", 7: "Databases engineer",
    8: "Devops Engineer", 9: "Dotnet Developer", 10: "ETL Developer", 11: "Electrical Engineer",
    12: "HR", 13: "Hadoop developer/Engineer", 14: "Health and fitness", 15: "Java Developer",
    16: "Mechanical Engineer", 17: "Network security engineer", 18: "Operations Manager",
    19: "PMO", 20: "Python Developer", 21: "SAP Developer", 22: "Business development executive",
    23: "Tester", 24: "Web Designing"
}

# Get readable job category
category_name = category_mapping.get(prediction_id, "Unknown")

# Print and log the prediction
print("Analysed category is:", category_name)
log_prediction(resume_path, category_name)

# Print prediction ID for debugging
print(prediction_id)

