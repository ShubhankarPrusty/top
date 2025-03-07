import sys
import threading
import os
import pandas as pd
import wave
import numpy as np
import librosa
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QLineEdit, QLabel, QComboBox, QSpacerItem, QSizePolicy,
                             QInputDialog, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from deep_translator import GoogleTranslator
import speech_recognition as sr
from speechbrain.inference import SpeakerRecognition
import soundfile as sf
import glob
from pydub import AudioSegment

class AudioProcessor:
    def __init__(self, dataset_dir, mfcc_limit=50):
        self.dataset_dir = dataset_dir
        self.mfcc_limit = mfcc_limit
        self.audio_samples_dir = os.path.join(self.dataset_dir, "audio_samples")
        os.makedirs(self.audio_samples_dir, exist_ok=True)

    def get_audio_path_for_word(self, word):
        data_csv = os.path.join(self.dataset_dir, "data.csv")
        df = pd.read_csv(data_csv)
        audio_file = df[df['correct_text'] == word]['audio_file'].values
        if len(audio_file) > 0:
            return os.path.join(self.audio_samples_dir, audio_file[0])
        return None

    def calculate_stddev(self, mfcc1, mfcc2):
        return np.std(mfcc1 - mfcc2)

    def build_dataset_mfcc(self):
        dataset_mfccs = {}
        data_csv = os.path.join(self.dataset_dir, "data.csv")
        df = pd.read_csv(data_csv)
        for _, row in df.iterrows():
            word = row["correct_text"]
            audio_path = os.path.join(self.audio_samples_dir, row["audio_file"])
            if os.path.exists(audio_path):
                mfcc = self.get_mfcc(audio_path)
                if word not in dataset_mfccs:
                    dataset_mfccs[word] = []
                dataset_mfccs[word].append(mfcc)
        return dataset_mfccs

    def get_stddev_for_all_words(self, original_audio_path):
        
        original_mfcc = self.get_mfcc(original_audio_path)
        dataset_mfccs = self.build_dataset_mfcc()
        stddev_values = []
        
        for word, mfcc_list in dataset_mfccs.items():
            for mfcc in mfcc_list:
                stddev = self.calculate_stddev(original_mfcc, mfcc)
                stddev_values.append((word, stddev))
        
        return stddev_values


class TrainingThread(QThread):
    training_complete = pyqtSignal(str)

    def __init__(self, audio_file_path, training_data_file, correct_text):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.training_data_file = training_data_file
        self.correct_text = correct_text
        self.dataset_dir = "dataset"
        os.makedirs(self.dataset_dir, exist_ok=True)  

    def run(self):
        self.save_training_data(self.correct_text)

    def save_training_data(self, correct_text):
        try:
            audio_filename = self.get_audio_filename(correct_text)
            os.rename(self.audio_file_path, audio_filename)

            if os.path.exists(self.training_data_file):
                df = pd.read_csv(self.training_data_file)
            else:
                df = pd.DataFrame(columns=["correct_text", "audio_file"])

            new_entry = pd.DataFrame({"correct_text": [correct_text], "audio_file": [audio_filename]})
            df = pd.concat([df, new_entry], ignore_index=True)

            df.to_csv(self.training_data_file, index=False)
            self.training_complete.emit(f"Training Data Saved: {correct_text} -> {audio_filename}")
        except Exception as e:
            self.training_complete.emit(f"Error saving training data: {str(e)}")

    
    def get_audio_filename(self, correct_text):
        safe_text = ''.join(e for e in correct_text if e.isalnum() or e in (' ', '_')).rstrip()
        base_filename = os.path.join(self.dataset_dir, f"{safe_text}")
        counter = 1
        while os.path.exists(f"{base_filename}_{counter}.wav"):
            counter += 1
        return f"{base_filename}_{counter}.wav"


class TranslatorChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_dir = "dataset"
        self.audio_processor = AudioProcessor(self.dataset_dir)
        self.initUI()
        self.recognizer = sr.Recognizer()
        self.voice_text = ""
        self.training_enabled = False
        self.training_data_file = "data.csv"
        self.recording = False  


    def start_recording(self):
        audio_file_path = os.path.join(self.audio_processor.audio_samples_dir, "audio_temp.wav")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=3)
            with wave.open(audio_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(audio.get_wav_data())
            self.voice_text = self.recognizer.recognize_google(audio)
            self.chat_display.append(f"<b>You (voice):</b> {self.voice_text}")
            self.message_input.setText(self.voice_text)

        
            recognized_audio_filename = os.path.join(self.audio_processor.audio_samples_dir, f"{self.voice_text}.wav")
            os.rename(audio_file_path, recognized_audio_filename)
            subparts = self.audio_processor.split_audio_into_subparts(recognized_audio_filename, self.voice_text)

    def start_recording(self):
        audio_file_path = os.path.join(self.dataset_dir, "audio_temp.wav")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=3)
            with wave.open(audio_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(audio.get_wav_data())
            self.voice_text = self.recognizer.recognize_google(audio)
            self.chat_display.append(f"<b>You (voice):</b> {self.voice_text}")
            self.message_input.setText(self.voice_text)


    def start_recording(self):
        audio_file_path = os.path.join(self.dataset_dir, "audio_temp.wav")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=3)
            with wave.open(audio_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(audio.get_wav_data())
            self.voice_text = self.recognizer.recognize_google(audio)
            self.chat_display.append(f"<b>You (voice):</b> {self.voice_text}")
            self.message_input.setText(self.voice_text)
            
            recognized_audio_filename = os.path.join(self.audio_processor.audio_samples_dir, f"{self.voice_text}.wav")
            os.rename(audio_file_path, recognized_audio_filename)
            
            subparts = self.audio_processor.split_audio_into_subparts(recognized_audio_filename, self.voice_text)
            if self.training_enabled:
                self.prompt_correct_word(recognized_audio_filename)


    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.voice_button.setText("Recording...")
            self.voice_button.setStyleSheet("""
                background-color: #27ae60;
                color: white;
                border-radius: 10px;
                padding: 10px;
                border: none;
            """)
            self.chat_display.append("<b>Listening...</b>")
            threading.Thread(target=self.start_recording, daemon=True).start()
        else:
            self.stop_recording()

    def stop_recording(self):
        self.recording = False
        self.voice_button.setText("Speak")
        self.voice_button.setStyleSheet("""        
            background-color: #e74c3c;
            color: white;
            border-radius: 10px;
            padding: 10px;
            border: none;
        """)


    def prompt_correct_word(self, audio_file_path):
        word, ok = QInputDialog.getText(self, "Correct the Word", "Please enter the correct word:")
        if ok and word:
            self.training_thread = TrainingThread(audio_file_path, self.training_data_file, word)
            self.training_thread.training_complete.connect(self.display_training_message)
            self.training_thread.start()
            

    def swap_selected_word(self):
        cursor = self.chat_display.textCursor()
        selected_text = cursor.selectedText().strip()

        if selected_text:
        
            selected_audio_path = os.path.join(self.audio_processor.audio_samples_dir, f"{selected_text}.wav")
        
        
            if os.path.exists(selected_audio_path):
                selected_mfcc = self.audio_processor.get_mfcc(selected_audio_path)
                dataset_mfccs = self.audio_processor.build_dataset_mfcc()
            
          
                matches = self.audio_processor.get_best_matches(selected_mfcc, dataset_mfccs)
                if matches:
               
                    best_match_word = matches[0][0]
                    corrected_audio_path = self.audio_processor.get_audio_path_for_word(best_match_word)
                
                    if corrected_audio_path and os.path.exists(corrected_audio_path):
                   
                        self.last_translated_text = self.last_translated_text.replace(selected_text, best_match_word)
                        self.chat_display.append(f"<b>Swapped:</b> {self.last_translated_text}")
                    else:
                        self.chat_display.append(f"<b>No audio file found for '{best_match_word}' in dataset.</b>")
                else:
                    self.chat_display.append("<b>No matches found for swapping.</b>")
            else:
                self.chat_display.append(f"<b>No audio file found for '{selected_text}' in audio_samples.</b>")
        else:
            self.chat_display.append("<b>Please select a word to swap.</b>")



    def toggle_training(self, state):
        self.training_enabled = (state == Qt.Checked)

    def display_training_message(self, message):
        self.chat_display.append(f"<b>{message}</b>")

    
    def get_audio_path_for_word(self, word):
        data_csv = os.path.join(self.dataset_dir, "data.csv")
        df = pd.read_csv(data_csv)
        audio_file = df[df['correct_text'] == word]['audio_file'].values
        return audio_file[0] if len(audio_file) > 0 else None  

    def initUI(self):
        self.setWindowTitle("Real-Time Translator Chat with Voice Input")
        self.setGeometry(100, 100, 800, 800)
        self.setStyleSheet("background-color: #2e2e2e;")
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)

        title_label = QLabel("Real-Time Translator Chat with Voice Input")
        title_label.setFont(QFont('Arial', 20))
        title_label.setStyleSheet("color: #f4f4f4;")
        title_label.setAlignment(Qt.AlignCenter)

        self.chat_display = QTextEdit(self)
        self.chat_display.setFont(QFont('Arial', 12))
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""        
            background-color: #1e1e1e;
            color: #f4f4f4;
            border-radius: 10px;
            padding: 10px;
            border: 2px solid #5c5c5c;
        """)
        self.chat_display.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_display.customContextMenuRequested.connect(self.show_context_menu)

        input_layout = QHBoxLayout()

        self.message_input = QLineEdit(self)
        self.message_input.setFont(QFont('Arial', 14))
        self.message_input.setStyleSheet("""        
            background-color: #3c3c3c;
            color: #f4f4f4;
            border-radius: 10px;
            padding: 10px;
            border: 2px solid #5c5c5c;
        """)

        send_button = QPushButton("Send", self)
        send_button.setFont(QFont('Arial', 14))
        send_button.setStyleSheet("""        
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px;
            border: none;
        """)
        send_button.clicked.connect(self.send_message)

        self.voice_button = QPushButton("Speak", self)
        self.voice_button.setFont(QFont('Arial', 14))
        self.voice_button.setStyleSheet("""        
            background-color: #e74c3c;
            color: white;
            border-radius: 10px;
            padding: 10px;
            border: none;
        """)
        self.voice_button.clicked.connect(self.toggle_recording)

        input_layout.addWidget(self.message_input)
        input_layout.addWidget(send_button)
        input_layout.addWidget(self.voice_button)

        lang_layout = QHBoxLayout()

        lang_label = QLabel("Translate to:")
        lang_label.setFont(QFont('Arial', 12))
        lang_label.setStyleSheet("color: #f4f4f4;")

        self.lang_combo = QComboBox(self)
        self.lang_combo.setFont(QFont('Arial', 12))
        self.lang_combo.setStyleSheet("""        
            background-color: #3c3c3c;
            color: #f4f4f4;
            border-radius: 5px;
            padding: 5px;
        """)

      
        self.swap_button = QPushButton("Swap Selected Word", self)
        self.swap_button.setFont(QFont('Arial', 14))
        self.swap_button.setStyleSheet("""        
            background-color: #3498db;
            color: white;
            border-radius: 10px;
            padding: 10px;
            border: none;
        """)
        self.swap_button.clicked.connect(self.swap_selected_word)

       
        main_layout.addWidget(self.swap_button)

        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("Spanish", "es")
        self.lang_combo.addItem("French", "fr")
        self.lang_combo.addItem("German", "de")

        self.swap_button = QPushButton("Swap Word with Stddev", self)
        self.swap_button.setFont(QFont('Arial', 14))
        self.swap_button.setStyleSheet("""        
            background-color: #3498db;
            color: white;
            border-radius: 10px;
            padding: 10px;
            border: none;
        """)
        self.swap_button.clicked.connect(self.prompt_for_word_swap)


        self.training_checkbox = QCheckBox("Enable Training Mode", self)
        self.training_checkbox.setStyleSheet("color: #f4f4f4;")
        self.training_checkbox.stateChanged.connect(self.toggle_training)

        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        lang_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding))
        lang_layout.addWidget(self.training_checkbox)

        main_layout.addWidget(title_label)
        main_layout.addWidget(self.chat_display)
        main_layout.addLayout(input_layout)
        main_layout.addLayout(lang_layout)
        self.setLayout(main_layout)

    def show_context_menu(self, pos):
        context_menu = QMenu(self)
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_to_clipboard)
        context_menu.addAction(copy_action)
        context_menu.exec_(self.chat_display.mapToGlobal(pos))

    def copy_to_clipboard(self):
        cursor = self.chat_display.textCursor() 
        selected_text = cursor.selectedText() if cursor.hasSelection() else self.chat_display.toPlainText() 
        clipboard = QApplication.clipboard() 
        clipboard.setText(selected_text) 
        self.chat_display.append("<b>Copied to clipboard!</b>") 

    def get_audio_path_for_word(self, word):
        data_csv = os.path.join(self.dataset_dir, "data.csv")
        df = pd.read_csv(data_csv)
        audio_file = df[df['correct_text'] == word]['audio_file'].values
        return audio_file[0] if len(audio_file) > 0 else None 

    
    def send_message(self):
        text = self.message_input.text().strip()
        target_lang = self.lang_combo.currentData()
    
        if not text:
            self.chat_display.append("<b>Error:</b> No text to translate.")
            return

        try:
            translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
            self.chat_display.append(f"<b>You:</b> {text}")
            self.chat_display.append(f"<b>Translated:</b> {translated_text}")
            self.message_input.clear()
            self.last_translated_text = translated_text
        except Exception as e:
            self.chat_display.append("<b>Error:</b> Translation failed. Please check your connection.")

        for file_path in glob.glob(os.path.join(self.audio_processor.audio_samples_dir, "*.wav")):
            os.remove(file_path)
    def prompt_for_word_swap(self):
        """Prompt for word swapping, show dropdown with stddev values."""
        try:
            original_audio_path = os.path.join(self.dataset_dir, "audio_temp.wav")
        
        # Check if the file exists before proceeding
            if not os.path.exists(original_audio_path):
                self.chat_display.append("<b>Error:</b> Original audio file not found.")
                return
        
        # Get standard deviation values for all words
            stddev_values = self.audio_processor.get_stddev_for_all_words(original_audio_path)
        
        # If no words found, show an error
            if not stddev_values:
                self.chat_display.append("<b>Error:</b> No words found for swapping.")
                return
        
        # Prepare word and stddev pairs to display in the dropdown
            word_stddev_pairs = [f"{word} (stddev: {stddev:.4f})" for word, stddev in stddev_values]

        # Create the input dialog
            dialog = QInputDialog(self)
            dialog.setWindowTitle("Select Word to Swap")
            dialog.setLabelText("Choose a word:")
            dialog.setComboBoxItems(word_stddev_pairs)
        
        # Set dialog size and center it on the screen
            dialog.setFixedSize(400, 150)  # Adjust the size if needed
            screen_geometry = QApplication.primaryScreen().geometry()
            dialog.move(screen_geometry.center() - dialog.rect().center())

        # Show dialog and handle result
            if dialog.exec_() == QDialog.Accepted:
                word = dialog.textValue()

                if word:
                    # Extract the word from the dropdown string
                    self.selected_stddev_word = word.split(" (")[0]
                    self.message_input.setText(self.selected_stddev_word)
                    self.chat_display.append(f"<b>Swapped:</b> {self.selected_stddev_word}")
    
        except Exception as e:
            # If an error occurs, display it in the chat
            self.chat_display.append(f"<b>Error:</b> {str(e)}")

    
def main():
    app = QApplication(sys.argv)
    window = TranslatorChatApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 
