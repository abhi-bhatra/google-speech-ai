import html
import io
import queue
import re
import sys
from google.cloud import speech, texttospeech
import pyaudio
import google.generativeai as genai

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
genai.configure(api_key="AIzaSyDczS_6wk30rFWeUasrFF7IVlocLrV2NFI")

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int = RATE, chunk: int = CHUNK) -> None:
        """The audio -- and generator -- is guaranteed to be on the main thread."""
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self: object) -> object:
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        """Closes the stream, regardless of whether the connection was lost or not."""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
            in_data: The audio data as a bytes object
            frame_count: The number of frames captured
            time_info: The time information
            status_flags: The status flags

        Returns:
            The audio data as a bytes object
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Generates audio chunks from the stream of audio data in chunks.

        Args:
            self: The MicrophoneStream object

        Returns:
            A generator that outputs audio chunks.
        """
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def speechtotext(responses: object) -> str:
    """Iterates through server responses and returns the full transcript."""
    num_chars_printed = 0
    full_transcript = ""
    for response in responses:
        if not response.results:
            continue
        
        result = response.results[0]
        if not result.alternatives:
            continue
        
        transcript = result.alternatives[0].transcript
        
        overwrite_chars = " " * (num_chars_printed - len(transcript))
        
        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            full_transcript += transcript + " "
            num_chars_printed = 0
        
        if re.search(r"\b(बस|bye|exit)\b", transcript, re.I):
            print("Goodbye command recognized. Stopping...")
            return full_transcript.strip()
    
    return full_transcript.strip()

def complete_function() -> str:
    """Transcribe speech from audio file."""
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "hi-IN"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        model="command_and_search",
        use_enhanced=True,
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # Get the transcribed text
        transcribed_text = speechtotext(responses)
        
        if transcribed_text.strip():
            # Here you can call your ChatGPT API function with the transcribed text
            chatgpt_response = call_gemini_api(transcribed_text)
            return chatgpt_response
        else:
            return "No speech detected or transcribed."
    return "An error occurred during transcription."

def call_gemini_api(prompt: str) -> str:
    """Mock function to call ChatGPT API."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    print(response.text)
    full_text = response.text
    words = full_text.split()
    trimmed_text = ' '.join(words[:50])
    return trimmed_text

def text_to_speech_stream(text: str) -> None:
    """
    Converts plaintext to SSML, generates synthetic audio from SSML,
    and plays it directly through the speakers.

    Args:
    text (str): text to synthesize and play

    Returns:
    None
    """
    # Replace special characters with HTML Ampersand Character Codes
    escaped_lines = html.escape(text)

    # Convert plaintext to SSML
    ssml = "<speak>{}</speak>".format(
        escaped_lines.replace("\n", '\n<break time="1s"/>')
    )

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Sets the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

    # Builds the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code="hi-IN", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Selects the type of audio file
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Performs the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Play the audio
    audio_data = io.BytesIO(response.audio_content)
    
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2),  # 16-bit
                    channels=1,
                    rate=24000,
                    output=True)

    chunk = 1024
    data = audio_data.read(chunk)

    while data:
        stream.write(data)
        data = audio_data.read(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Audio playback completed.")

def main() -> None:
    """Main function to run the speech-to-text and text-to-speech pipeline."""
    response = complete_function()
    print(response)
    if response != "No speech detected or transcribed." and response != "An error occurred during transcription.":
        text_to_speech_stream(response)
    else:
        print("No audio detected or transcribed. Exiting...")

if __name__ == "__main__":
    main()
