Offline transcriber based on open-weight models: transcribes, summarizes, creates bulleted lists. Compatible with all audio formats.

# Description
If you are interested in a completely *private* solution for transcribing audio recordings, creating summaries and outlines of what has been transcribed, and being able to work even when you are not connected to the internet, this simple program is the answer!

Simply download the OpenAI [Whisper large v3](https://huggingface.co/openai/whisper-large-v3) model from Huggin Face to the `whisper-large-v3` directory, then install `Ollama` on your computer and use the `ollama pull ...` command to request the download of an open-weight model of your choice, for example `qwen3:30b-a3b-instruct-2507-q4_K_M` which can also operate with a low amount of available VRAM.

# Details
This is a Gradio web app with the following features:

- Upload of one or more audio files; optional conversion to 16 kHz mono WAV if the file is in a different format (mp3, etc.)  using ffmpeg (must be installed) and temporary file management.
- Initialization of the Whisper model (whisper-large-v3) from a local folder with PyTorch/Transformers and automatic use of GPU if available.
- Sequential transcription of all uploaded files.
- Creation of summary and bullet list via local Ollama API (`http://localhost:11434`) using the model specified in `OLLAMA_MODEL`.
- HTML rendering of results with cards and buttons to copy text to the clipboard; bilingual IT/EN UI with custom CSS and JS styling.
- No external services required, completely offline once the necessary models have been downloaded.

# Usage

Create a Python environment (`python3 -m venv myenv`), install the requirements (`pip install -r requirements.txt`), and simply run the program. You can then use the application by going to the address shown in the terminal with a browser, for example:

`* Running on local URL:  http://127.0.0.1:49591`

<img width="100%" alt="example" style="width:100%" src="https://github.com/user-attachments/assets/60bcbff3-c89f-4d75-be14-4cc651c0b80e" />

# Modification / customization

At the beginning of the program, all the variables are present to customize the program, add languages, or modify prompts for Ollama.

<img width="100%" alt="customization" style="width:100%" src="https://github.com/user-attachments/assets/5cc1d366-3177-4532-8ff0-39c9a76a8518" />

