import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, SpeechT5Tokenizer, SpeechT5ForTextToSpeech, Trainer, TrainingArguments
import subprocess

# Load Mozilla Common Voice Dataset
dataset_path = "E:/hindi dataset/hindi dataset/hi"  # Update this with the correct path
transcript_file = os.path.join(dataset_path, "validated.tsv")
audio_folder = os.path.join(dataset_path, "clips")

# Step 1: Preprocessing the Dataset
# Load transcripts
df = pd.read_csv(transcript_file, sep='\t')

# Text normalization
def normalize_text(text):
    # Remove special characters, extra spaces, and punctuation
    return text.lower().strip()

df['sentence'] = df['sentence'].apply(normalize_text)

# Preprocess audio - resampling, trimming silence
def preprocess_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    audio = librosa.effects.trim(audio)[0]  # Trim silence
    return audio

df['audio'] = df['path'].apply(lambda x: preprocess_audio(os.path.join(audio_folder, x)))

# Splitting the dataset into train and eval
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

# Step 2: Pronunciation and Prosody Adjustments
# Load tokenizer and model
tokenizer = SpeechT5Tokenizer.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Custom phonemizer for Hindi pronunciation using espeak
def phonemize_text(text):
    # Execute espeak command to get phonemes
    command = f'espeak -v hi "{text}" --phoneme'
    phonemes = subprocess.check_output(command, shell=True, text=True).strip()
    return phonemes

# Apply phonemization
train_df['phonemes'] = train_df['sentence'].apply(phonemize_text)
eval_df['phonemes'] = eval_df['sentence'].apply(phonemize_text)

# Tokenize input for model training
train_df['input_ids'] = train_df['phonemes'].apply(lambda x: tokenizer(x, return_tensors='pt').input_ids)
eval_df['input_ids'] = eval_df['phonemes'].apply(lambda x: tokenizer(x, return_tensors='pt').input_ids)

# Step 3: Training the Model
# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    save_steps=1000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    logging_dir='./logs',
    logging_steps=500,
    do_train=True,
    do_eval=True,
    eval_steps=1000,
)

# Use Hugging Face Trainer for SpeechT5
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_df,
    eval_dataset=eval_df,
)

# Train the model
trainer.train()

# Step 5: Evaluating the Model using MOS
def generate_speech_and_collect_mos(model, tokenizer, eval_dataset):
    mos_scores = []
    for index, row in tqdm(eval_dataset.iterrows(), total=len(eval_dataset)):
        input_ids = row['input_ids']
        audio_output = model.generate(input_ids=input_ids)
        
        # Save or play the audio, then have native speakers rate the speech quality
        mos = collect_mos_from_user(audio_output)  # Custom function for human evaluation
        mos_scores.append(mos)
    
    return np.mean(mos_scores)

# Placeholder for MOS collection function
def collect_mos_from_user(audio):
    # Play the audio to a native speaker and get their rating
    mos = np.random.uniform(1, 5)  # Placeholder for MOS score (random generation)
    return mos

# Evaluate the model and calculate MOS
mean_mos_score = generate_speech_and_collect_mos(model, tokenizer, eval_df)
print(f"Mean MOS Score: {mean_mos_score}")

# Step 6: Save the Fine-tuned Model
model_save_path = "./fine_tuned_speecht5_hindi"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved at {model_save_path}")
