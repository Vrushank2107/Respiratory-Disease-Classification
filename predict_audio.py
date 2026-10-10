"""Compatibility CLI for the migrated saved Lightweight Mel CNN inference."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
from app.backend.services.models import ROOT, load_signal, predict_dl

def predict(audio_path):
 y,source_sr,channels=load_signal(audio_path)
 result=predict_dl('lightweight_mel_cnn',y)
 print('\nResearch model output — not a medical diagnosis')
 print(f'Audio file: {audio_path}')
 print(f'Predicted class: {result["predicted_class"]}')
 print(f'Model score: {max(result["scores"].values()):.4f}')
 print('\nClass probabilities:')
 for name,score in sorted(result['scores'].items(),key=lambda x:x[1],reverse=True): print(f'  {name:20s} {score:.4f}')
 print('\nNote: these are softmax scores; calibration is not established.')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description='Run saved Lightweight Mel CNN on WAV audio; research use only.')
 parser.add_argument('audio_path'); predict(parser.parse_args().audio_path)
