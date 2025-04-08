from flask import Flask, request
from gtts import gTTS
import os

app = Flask(__name__)

def format_speech(steer, throttle):
    steer_deg = abs(steer * 25)
    direction = "Right" if steer > 0 else "Left" if steer < 0 else "Straight"
    speed = throttle * 30
    motion = "Accelerate" if throttle > 0.1 else "Brake"

    if direction == "Straight":
        return f"{motion} {speed:.1f} kilometers per hour"
    else:
        return f"{direction} {steer_deg:.1f} degrees, {motion} {speed:.1f} kilometers per hour"

@app.route("/say", methods=["POST"])
def speak_action():
    data = request.get_json()
    steer = float(data.get("steer", 0.0))
    throttle = float(data.get("throttle", 0.0))

    sentence = format_speech(steer, throttle)
    print(f"[AUDIO] {sentence}")

    tts = gTTS(text=sentence, lang='en')
    tts.save("action.mp3")
    os.system("mpg321 action.mp3")  # You can replace with 'afplay' or 'omxplayer' based on OS

    return {"status": "spoken"}

if __name__ == "__main__":
    app.run(port=5005)