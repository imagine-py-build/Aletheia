"""Download Aletheia's inference models into the persistent Hugging Face cache.

This is optional: inference also downloads models lazily on first use. Because
HF_HOME is mounted to the persistent model cache, repeated Docker rebuilds do
not download another copy.
"""
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor, AutoModelForAudioClassification

IMAGE_MODEL = 'buildborderless/CommunityForensics-DeepfakeDet-ViT'
AUDIO_MODEL = 'Vansh180/deepfake-audio-wav2vec2'
print('HF_HOME:', os.getenv('HF_HOME', '<default>'))

print('Checking image model cache:', IMAGE_MODEL)
AutoImageProcessor.from_pretrained(IMAGE_MODEL)
AutoModelForImageClassification.from_pretrained(IMAGE_MODEL)
print('Image model ready/cached.')

print('Checking audio model cache:', AUDIO_MODEL)
AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL)
print('Audio model ready/cached.')
