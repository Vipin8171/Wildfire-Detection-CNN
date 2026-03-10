# Phase 5: Deployment & Web Application

## Flask Web Application

### Features
- **Upload Interface**: Drag & drop satellite patch files (.npy / .npz)
- **Demo Mode**: Test with random samples from the dataset
- **Prediction Visualization**: Fire probability map + binary mask
- **Feature Analysis**: All 12 channels displayed with statistics
- **Risk Assessment**: LOW / MEDIUM / HIGH based on fire coverage

### Endpoints
| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main upload page |
| `/about` | GET | Project information |
| `/predict` | POST | Upload file and get prediction |
| `/demo` | POST | Run demo on random test sample |

### Input Format
- `.npy` file: shape (12, 64, 64) or (64, 64, 12)
- `.npz` file: with key "patch", "data", "X", or first key

### Output
- Fire probability heatmap
- Binary fire mask prediction
- 12-channel feature visualization
- Feature statistics table
- Risk level (LOW/MEDIUM/HIGH)
- Detailed JSON report

### Running
```bash
python webapp/app.py
# Open http://localhost:5000
```
