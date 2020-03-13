import os
import json

from TTS.datasets.preprocess import get_preprocessor_by_name


def make_speakers_json_path(out_path):
    """Returns conventional speakers.json location."""
    return os.path.join(out_path, "speakers.json")


def load_speaker_mapping(out_path):
    """Loads speaker mapping if already present."""
    try:
        if os.path.splitext(out_path) == '.json':
            json_file = out_path
        else: 
            json_file = make_speakers_json_path(out_path)
        with open(json_file) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_speaker_mapping(out_path, speaker_mapping):
    """Saves speaker mapping if not yet present."""
    speakers_json_path = make_speakers_json_path(out_path)
    with open(speakers_json_path, "w") as f:
        json.dump(speaker_mapping, f, indent=4)


def get_speakers(items):
    """Returns a sorted, unique list of speakers in a given dataset."""
    speakers = {e[2] for e in items}
    return sorted(speakers)


def get_speakers_embedding(speaker_mapping):
    """Returns the list of embeddings of all speakers present on speaker_mapping"""
    return [speaker_mapping[speaker]['embedding'] for speaker in speaker_mapping]


def get_speakers_id(speaker_mapping,speaker_names):
    """Returns the list of ids of all speakers present on speakers_names"""
    return [speaker_mapping[speaker_name]['id'] for speaker_name in speaker_names]       