from pathlib import Path
class SpeakerVerifier:
    def __init__(self,model_dir):
        if not Path(model_dir).exists(): raise FileNotFoundError(f'Speaker model directory not found: {model_dir}')
        self.model_dir=model_dir
    def compare(self,reference,suspected):
        raise NotImplementedError('Use a licensed ECAPA-TDNN/SpeechBrain checkpoint; Aletheia does not fabricate identity similarity.')
