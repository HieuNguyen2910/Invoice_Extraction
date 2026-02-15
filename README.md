# Invoice Extraction System
Automatically extracts key information such as company name, date, address, and total amount from invoice images using OCR and layout-aware deep learning models.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [License](#license)

## Installation

1. Clone the repository

```bash
git clone https://github.com/HieuNguyen2910/Invoice_Extraction
cd Invoice_Extraction
```

2. Create and activate Conda environment

```bash
conda create -n your_env python=3.8 -y
conda activate your_env
```

3. Install required dependencies
```bash
pip install -r requirements.txt
```

## Usage

Download the pretrained model weights.

🔗 **Weights download link:**  
*(https://drive.google.com/drive/u/8/folders/14UvUr8fxxmkPFlnTlX9RN8RCYsFvM-8y)*

Make sure the file paths in the source code match the downloaded weights.

Run the Django Development Server
```bash
python manage.py runserver
```

Access the Web Application: 
open your browser and go to 
*(http://127.0.0.1:8000/)*

## Demo

Below is an example of Face Attendance on a face.

![Demo Result](assets/demo.jpg)





