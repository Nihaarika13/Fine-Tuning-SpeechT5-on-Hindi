# Fine-Tuning-SpeechT5-on-Hindi


This project focuses on fine-tuning a Text-to-Speech (TTS) model for Hindi language speech synthesis using the SpeechT5 architecture. The model is trained on the Mozilla Common Voice dataset to generate natural-sounding speech from Hindi text input.

MOS Value: 4.173

Link for dataset: https://commonvoice.mozilla.org/en/datasets
Choose "Hindi" in the languages

Python 3.7 or higher(prefer 3.11.10)
- [PyTorch](https://pytorch.org/get-started/locally/) (with the correct version based on your CUDA setup)
- Other dependencies specified in `requirements.txt`

# Fine-Tuning SpeechT5 for Hindi Text-to-Speech
This repository provides the steps and code to fine-tune the SpeechT5 model for Hindi text-to-speech (TTS) using the Mozilla Common Voice dataset. The guide walks through dataset preparation, audio preprocessing, phonemization using eSpeak, and model training.

Prerequisites
Ensure that you have the following installed on your system:

->Python 3.7 or higher(prefer 3.11.10)

->GPU (optional but highly recommended for faster training)

->Git

Installation
Follow the steps below to set up the environment and install the required dependencies.

Step 1: Clone the Repository
First, clone this repository to your local machine:

Code:

           git clone https://github.com/your-repo/Fine-Tuning-SpeechT5-on-Hindi.git
           cd Fine-Tuning-SpeechT5-on-Hindi

Step 2: Set Up a Virtual Environment (Optional)
It's recommended to use a virtual environment to isolate dependencies:

Code:

        python -m venv venv
        source venv/bin/activate  # On Linux/macOS
        
 OR
 
        venv\Scripts\activate      # On Windows
        
Step 3: Install Requirements
The required Python packages are listed in the requirements.txt file. Install them using the following command:

Code:

        pip install -r requirements.txt

You may also need to install additional packages like espeak for phonemization:

Step 4: Install eSpeak for Phonemization
For phonemization (converting Hindi text to phonemes), we use eSpeak. Install eSpeak with the following commands:

Linux:
Code

         sudo apt-get install espeak

Windows:

Download the installer from eSpeak's official site and follow the installation instructions.

Step 5: Verify Installations
After installing the dependencies, run the following command to verify if everything is correctly installed:

Code:

          python -c "import torch, librosa, pandas, transformers"
         
If no errors are raised, you are ready to proceed.
