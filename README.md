# Methods5_Artificial_Intelligence_and_Advanced_Analytics

## Emotion Release Emoji Machine

This project is a Streamlit web app for emotion analysis.

Users can:
- enter text directly
- upload an audio file
- get an emotion prediction
- see confidence scores
- get emoji-style emotional feedback

For audio input, the app first uses Whisper to transcribe speech into text, then analyzes the emotion of the transcription.

## Project files

- `app.py` — main Streamlit web app
- `emotion_app.py` — emotion prediction and audio transcription functions
- `Emotion_classify_Data.csv` — dataset
- `style.css` — app styling
- `emotion_app.ipynb` — development notebook
- `evaluate_model.ipynb` — model evaluation notebook
- `README.md` — project instructions

## Important: Whisper model file

The Whisper model file `base.pt` is not included in this GitHub repository because it is too large.

To download the Whisper base model, first install the dependencies:

```bash
pip install streamlit matplotlib openai-whisper transformers torch
```

Then run this command in Terminal:

```bash
python -c "import whisper; whisper.load_model('base')"
```

This command downloads the Whisper `base` model to your computer.

After the model is downloaded, create a folder named `whisper_models` in this project:

```bash
mkdir -p whisper_models
```

Then find the downloaded `base.pt` file on your computer and copy it into this folder.

The final path must be:

```text
whisper_models/base.pt
```

The final project structure should look like this:

```text
Methods5_Artificial_Intelligence_and_Advanced_Analytics/
├── app.py
├── emotion_app.ipynb
├── emotion_app.py
├── style.css
├── Emotion_classify_Data.csv
├── evaluate_model.ipynb
├── README.md
└── whisper_models/
    └── base.pt
```

Without `whisper_models/base.pt`, the audio transcription feature will not work.

## Run the app

Run this in Terminal:

```bash
streamlit run app.py
```

## Notes

The text emotion classifier will be downloaded automatically when the app runs.

The Whisper model file is not uploaded to GitHub, so it must be downloaded and placed manually before using the audio feature.