# Medicinal Plant Classifier - Backend Server

FastAPI server for medicinal plant classification using VECTVMixer model.

## Files

- `server.py` - FastAPI application
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration
- `labels.txt` - Plant class labels
- `best_vectvmixer.pth` - Trained PyTorch model

## Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check
- `POST /predict` - Plant classification (multipart form with image file)

## Local Testing

```bash
pip install -r requirements.txt
python server.py
```

Visit: http://localhost:8000/docs

## Render Deployment

1. Push to GitHub
2. Connect to Render
3. Deploy web service
4. Get API URL

## API Usage

```bash
curl -X POST "https://your-api.onrender.com/predict" \
  -F "file=@image.jpg"
```

Response:
```json
{
  "success": true,
  "predicted_class": "Aloe_barbadensis_miller",
  "confidence": 0.95,
  "all_predictions": {
    "Aloe_barbadensis_miller": 0.95,
    "Azadirachta_indica": 0.03,
    ...
  }
}
```
