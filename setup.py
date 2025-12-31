from setuptools import setup, find_packages

setup(
    name="bitvoice",
    version="0.1.0",
    description="A local, privacy-first Text-to-Speech tool supporting multiple engines and file formats.",
    author="BitVoice Team",
    py_modules=["bitvoice"],
    install_requires=[
        "kokoro-onnx",
        "soundfile",
        "pyttsx3",
        "gTTS",
        "piper-tts",
        "onnxruntime",
        "pypdf",
        "python-docx",
        "EbookLib",
        "beautifulsoup4",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "bitvoice=bitvoice:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
