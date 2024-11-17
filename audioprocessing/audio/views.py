import librosa
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import numpy as np
from archisound import ArchiSound
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_size):
        super(Classifier, self).__init__()
        self.Custom = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.Custom(x)
        return x

input_size= 7520
model = Classifier(input_size)
autoencoder = ArchiSound.from_pretrained("dmae1d-ATC64-v2")
model_path="model.pt"
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
model.eval()

def record_audio(request):
    return render(request, 'audio/record.html')

def process_audio(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']

        # Save the uploaded file
        fs = FileSystemStorage()
        file_path = fs.save(audio_file.name, audio_file)
        full_path = fs.path(file_path)
        # Load the audio file with librosa
        y, sr = librosa.load(full_path, duration=5)
        audio_features = {
            'sample_rate': sr,
            'duration': librosa.get_duration(y=y, sr=sr),
        }

        y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        sr = 48000
        out = call_model(y, sr)
        # Clean up saved file
        os.remove(full_path)
        print("out before return: ", out)
        print(type(out))
        if out == 0:
            return JsonResponse("The baby does not have RDS!", safe=False),
        else:
            return JsonResponse("The baby has RDS! Immidiately rush to the hospital. You can use our hospital finding tool.", safe=False)

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def call_model(y, sr):
    print("SR is: ", sr)
    if len(y) < 5*sr:
        y = np.pad(y, (5 * sr)-len(y), mode='constant')
    y = y[:5*sr]
    if y.ndim == 1:  # If mono, convert to stereo
        y = np.stack((y, y), axis=-1)
    z = torch.from_numpy(y).float()
    z = z.unsqueeze(0)
    z = z.permute(0, 2, 1)
    encoded= autoencoder.encode(z)
    out = model(encoded)
    out = out.detach().numpy()
    print(out)
    return int(out[0][0])