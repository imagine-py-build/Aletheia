# Aletheia model card

Aletheia is a framework, not a claim of universal deepfake detection accuracy. Task-specific detectors must be trained/evaluated on representative, source-separated datasets. The bundled architecture definitions are genuine PyTorch models. A detector without a task-trained checkpoint is intentionally unavailable at inference time.

## Human verification
Every AI result is marked separately from investigator verification. Reports explicitly state limitations.
