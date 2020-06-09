#!flask/bin/python
import argparse
import os

from flask import Flask, request, render_template, send_file
from TTS.server.synthesizer import Synthesizer


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ['true', '1', 'yes']

    parser = argparse.ArgumentParser()
    parser.add_argument('--tts_checkpoint', type=str, help='path to TTS checkpoint file')
    parser.add_argument('--tts_config', type=str, help='path to TTS config.json file')
    parser.add_argument('--tts_speakers', type=str, help='path to JSON file containing speaker ids, if speaker ids are used in the model')
    parser.add_argument('--wavernn_lib_path', type=str, help='path to WaveRNN project folder to be imported. If this is not passed, model uses Griffin-Lim for synthesis.')
    parser.add_argument('--wavernn_file', type=str, help='path to WaveRNN checkpoint file.')
    parser.add_argument('--wavernn_config', type=str, help='path to WaveRNN config file.')
    parser.add_argument('--is_wavernn_batched', type=convert_boolean, default=False, help='true to use batched WaveRNN.')
    parser.add_argument('--port', type=int, default=5002, help='port to listen on.')
    parser.add_argument('--use_cuda', type=convert_boolean, default=False, help='true to use CUDA.')
    parser.add_argument('--debug', type=convert_boolean, default=False, help='true to enable Flask debug mode.')
    return parser


config = None
synthesizer = None

tts_model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../tts_model')
tts_checkpoint_file = os.path.join(tts_model_folder, 'checkpoint_670000.pth.tar')
tts_config_file = os.path.join(tts_model_folder, 'config.json')

melgan_model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../melgan_model')
melgan_checkpoint_file = os.path.join(melgan_model_folder, 'checkpoint-1000000steps.pkl')
melgan_config_file = os.path.join(melgan_model_folder, 'config.yml')

if not os.path.isfile(tts_checkpoint_file):
  print("TTS model checkpoint file missing")
  exit(2)
if not os.path.isfile(tts_config_file):
  print("TTS config file missing")
  exit(2)
if not os.path.isfile(melgan_checkpoint_file):
  print("MelGAN model checkpoint file missing")
  exit(2)
if not os.path.isfile(melgan_config_file):
  print("MelGAN config file missing")
  exit(2)

# Use default config with embedded model files
config = create_argparser().parse_args([])
config.tts_checkpoint_file = tts_checkpoint_file
config.tts_config_file = tts_config_file
config.melgan_checkpoint_file = melgan_checkpoint_file
config.melgan_config_file = melgan_config_file
synthesizer = Synthesizer(config)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    print(" > Model input: {}".format(text))
    data = synthesizer.tts(text)
    return send_file(data, mimetype='audio/wav')


if __name__ == '__main__':
    args = create_argparser().parse_args()

    # Setup synthesizer from CLI args if they're specified or no embedded model
    # is present.
    if not config or not synthesizer or args.tts_checkpoint or args.tts_config:
        synthesizer = Synthesizer(args)

    app.run(debug=config.debug, host='127.0.0.1', port=config.port)
