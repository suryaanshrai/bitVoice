from bitvoice import BitVoice
import os

print("Testing library usage...")
try:
    bv = BitVoice(model="pyttsx3")
    bv.convert_text("Library integration successful.", "library_test.wav")
    if os.path.exists("library_test.wav"):
        print("Success: Audio generated.")
    else:
        print("Failure: Audio not generated.")
except Exception as e:
    print(f"Error: {e}")
