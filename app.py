import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH  = 'mobilenetv2_emotion_model.pth'
IMG_SIZE    = 96
NUM_CLASSES = 7
EMOTIONS    = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_EMOJI = {
    'angry':    '😠',
    'disgust':  '🤢',
    'fear':     '😨',
    'happy':    '😊',
    'neutral':  '😐',
    'sad':      '😢',
    'surprise': '😲',
}

# ── Model definition (must match training) ───────────────────────────────────
class EmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        base.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(1280),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )
        self.features   = base.features
        self.classifier = base.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── Load model once at startup ────────────────────────────────────────────────
print(f'Loading model on {DEVICE} …')
model = EmotionNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print('Model ready!')

# ── Preprocessing (same as test_tf in training) ───────────────────────────────
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # (1, 3, 96, 96)

        with torch.no_grad():
            logits = model(tensor)                          # (1, 7)
            probs  = torch.softmax(logits, dim=1)[0]       # (7,)

        top_idx   = probs.argmax().item()
        top_label = EMOTIONS[top_idx]
        top_prob  = probs[top_idx].item()

        all_probs = [
            {
                'emotion': EMOTIONS[i],
                'emoji':   EMOTION_EMOJI[EMOTIONS[i]],
                'prob':    round(probs[i].item() * 100, 2),
            }
            for i in range(NUM_CLASSES)
        ]
        all_probs.sort(key=lambda x: x['prob'], reverse=True)

        return jsonify({
            'prediction': top_label,
            'emoji':      EMOTION_EMOJI[top_label],
            'confidence': round(top_prob * 100, 2),
            'all_probs':  all_probs,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # host=0.0.0.0 so you can access from outside the VM
    app.run(host='0.0.0.0', port=5000, debug=False)
