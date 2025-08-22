
### 🔊 Text-to-Speech (TTS)

    ```
    from boo import TTS

    tts = TTS()
    outfile = tts.save_audio("Hello from Boo in a calm voice.", "out/hello.mp3")
    print("Saved:", outfile)
    ```

### 🎙️ Transcription / Translation (Whisper)

    ```
    from boo import Transcription, Translation

    asr = Transcription()
    text = asr.transcribe("audio/meeting.m4a")
    print(text)

    xlat = Translation()
    english = xlat.create("Translate this speech to English.", "audio/spanish.m4a")
    print(english)
    ```

