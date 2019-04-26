from setuptools import setup, find_packages

# read version from version.py
__version__ = None
with open("version.py") as f:
    exec(f.read())

setup_requires = ["numpy==1.15.4"],

install_requires = [
    "scipy >=0.19.0",
    "torch >= 0.4.1",
    "librosa==0.6.2",
    "unidecode==0.4.20",
    "tensorboardX",
    "matplotlib==2.0.2",
    "Pillow",
    "flask",
    # "lws",
    "tqdm",
    "soundfile",
    "pip @ https://github.com/bootphon/phonemizer/archive/v1.0.1"
]

extras_require = {
    "bin": [
        "requests",
    ]
}

setup(
    name='TTS',
    version=__version__,
    url='https://github.com/mozilla/TTS',
    description='Text to Speech with Deep Learning',
    packages=find_packages(),
    license="Mozilla Public License 2.0",
    keywords="voice speech text-to-speech tts",
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require=extras_require
)
