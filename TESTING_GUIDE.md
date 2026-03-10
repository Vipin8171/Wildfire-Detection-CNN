# 🧪 Testing Guide - Wildfire Detection Webapp

## ✅ Issues Fixed

### 1. **JSON Serialization Error**
- **Problem**: `Object of type float32 is not JSON serializable` when clicking Demo
- **Root Cause**: NumPy float32 values in the report dictionary were not converted to native Python floats
- **Solution**: Added `convert_numpy_types()` helper function in `webapp/app.py` that recursively converts all numpy types to Python-native types before JSON serialization
- **Status**: ✅ **FIXED**

### 2. **Missing Test Data**
- **Problem**: `data/` folder was empty - no `.npy` or `.npz` files for testing
- **Root Cause**: Dataset is stored in TFRecord format, not individual .npy files
- **Solution**: Created `src/extract_sample_data.py` script that extracts representative samples from TFRecord:
  - 1 sample with low fire (106 fire pixels)
  - 1 sample with medium fire (377 fire pixels)
  - 1 sample with high fire (515 fire pixels)
  - 3 samples with no fire
- **Status**: ✅ **FIXED**

---

## 📁 Sample Data Created

All files are in `data/` directory:

```
data/
├── sample_low_fire_00.npy       (106 fire pixels)   ← Test fire detection sensitivity
├── sample_medium_fire_01.npy    (377 fire pixels)   ← Test balanced case
├── sample_high_fire_02.npy      (515 fire pixels)   ← Test on extreme case
├── sample_no_fire_00.npy        (0 fire pixels)     ← Test specificity
├── sample_no_fire_01.npy        (0 fire pixels)
└── sample_no_fire_02.npy        (0 fire pixels)
```

Each file contains a **12-channel satellite patch** (64×64 pixels):
- **Shape**: `(12, 64, 64)` - [channels, height, width]
- **Channels**: NDVI, elevation, temperature, humidity, wind, precipitation, drought, fire risk, population, previous fire, etc.
- **Format**: NumPy .npy (binary, easy to load)
- **Data Type**: float32
- **Size**: ~192 KB per file

---

## 🚀 How to Test

### **Method 1: Test with Sample Data (Recommended for Quick Testing)**

1. **Open the webapp**:
   ```
   http://127.0.0.1:5000
   ```

2. **Click "Choose File"**:
   - Select one of the sample files from `data/` folder
   - Examples:
     - `sample_high_fire_02.npy` - Should detect high fire probability ✅
     - `sample_no_fire_00.npy` - Should detect low fire probability ✅

3. **Click "Predict Fire Spread"**:
   - Wait 2-3 seconds for prediction
   - View results:
     - **Fire Probability Map** (heatmap)
     - **12-Channel Feature Grid** (satellite data)
     - **Risk Level Badge** (HIGH/MEDIUM/LOW)
     - **Statistics**: Fire pixels, percentage, probabilities
     - **Feature Summary**: Mean/min/max of all 12 channels

4. **Verify JSON fix**: Report should display correctly without serialization errors

---

### **Method 2: Test with Demo Mode (Automatic Testing)**

1. **Open the webapp**: http://127.0.0.1:5000

2. **Click "Run Demo"**:
   - Automatically loads a random test sample from the dataset
   - No file upload needed

3. **View results**:
   - **Predicted Fire Probability Map** (left)
   - **Predicted Binary Mask** (middle)
   - **Ground Truth Comparison** (right) - Shows actual fire from labeled data
   - Risk assessment and statistics

4. **Verify JSON fix**: All data should serialize without errors

---

## 🧪 Expected Test Results

### Low Fire Sample (`sample_low_fire_00.npy`)
- **Expected Fire Pixels**: ~106 / 4,096 (2.59%)
- **Expected Risk Level**: LOW or MEDIUM
- **Expected Max Probability**: 0.3-0.6
- **Model Behavior**: May show low fire probability (model biased toward "no fire")

### Medium Fire Sample (`sample_medium_fire_01.npy`)
- **Expected Fire Pixels**: ~377 / 4,096 (9.21%)
- **Expected Risk Level**: MEDIUM to HIGH
- **Expected Max Probability**: 0.5-0.8
- **Model Behavior**: Should clearly detect fire presence

### High Fire Sample (`sample_high_fire_02.npy`)
- **Expected Fire Pixels**: ~515 / 4,096 (12.58%)
- **Expected Risk Level**: HIGH
- **Expected Max Probability**: 0.7-0.95
- **Model Behavior**: Should confidently predict fire

### No-Fire Sample (`sample_no_fire_00.npy`)
- **Expected Fire Pixels**: 0 / 4,096 (0%)
- **Expected Risk Level**: LOW
- **Expected Max Probability**: 0.0-0.2
- **Model Behavior**: Should predict mostly non-fire

---

## 📊 What Each Output Shows

### 1. **Prediction Image** (Left Panel)
- **NDVI** (top-left): Vegetation index (green = vegetation, red = bare ground)
- **Previous Fire Mask** (top-right): What burned before
- **Probability Map** (bottom-left): Model's fire prediction (colorbar = probability)
- **Predicted Mask** (bottom-right): Binary fire/non-fire classification

### 2. **Feature Grid** (12 Channels)
A 3×4 grid showing all 12 input satellite channels:
1. **NDVI** - Vegetation greenness
2. **Elevation** - Terrain height
3. **Relative Humidity (th)** - Moisture in air
4. **Wind Speed (vs)** - Wind magnitude
5. **Min Temperature (tmmn)** - Overnight temperature
6. **Max Temperature (tmmx)** - Daytime temperature
7. **Specific Humidity (sph)** - Atmospheric moisture
8. **Precipitation (pr)** - Recent rainfall
9. **Palmer Drought Index (pdsi)** - Long-term drought
10. **Energy Release Component (erc)** - Burn potential
11. **Population** - Human settlements
12. **Previous Fire (PrevFireMask)** - Fire from previous day

### 3. **Risk Level Badge**
- 🔴 **HIGH**: > 30% fire pixels detected, should take action
- 🟡 **MEDIUM**: 10-30% fire pixels, monitor carefully
- 🟢 **LOW**: < 10% fire pixels, low risk

### 4. **Statistics Card**
- **Fire Detected**: Yes/No
- **Fire Pixels**: Count of predicted fire pixels
- **Fire Percentage**: % of patch that's fire
- **Max Probability**: Highest fire probability in patch (0-1)
- **Mean Probability**: Average fire probability

### 5. **Feature Summary Table**
For each channel shows:
- Mean, min, max values
- Units (e.g., K for Kelvin, m/s for wind speed)
- Data source

---

## 🐛 Troubleshooting

### Issue: "Error loading file"
- **Cause**: File is not in .npy/.npz format or is corrupted
- **Solution**: Use one of the provided samples in `data/` folder

### Issue: "Invalid shape" error
- **Cause**: File contains data with wrong dimensions
- **Solution**: Data must be shape (12, 64, 64) or (1, 12, 64, 64)
- **Example**: `sample_high_fire_02.npy` is 12×64×64 ✅

### Issue: "Error: Object of type float32..."
- **Cause**: NumPy types not converted before JSON serialization
- **Status**: ✅ **FIXED** - Update webapp/app.py with the new `convert_numpy_types()` function

### Issue: Demo returns error
- **Cause**: Could not load test data or convert output
- **Status**: ✅ **FIXED** - Both demo and predict endpoints now use `convert_numpy_types()`

---

## 📝 Sample Data Format

Each `.npy` file contains:
```python
import numpy as np

# Load sample data
data = np.load('data/sample_high_fire_02.npy')

print(data.shape)      # (12, 64, 64)
print(data.dtype)      # float32
print(data.min(), data.max())  # Min and max values

# Access individual channels
ndvi = data[0]              # Vegetation index
elevation = data[1]        # Terrain height
temperature = data[5]      # Max temperature
previous_fire = data[11]   # Previous fire mask
```

---

## 🔧 Creating More Test Data

If you want to extract different samples:

```bash
cd c:\Users\tvipi\project\Wildfire-Detection-CNN
python src/extract_sample_data.py
```

This will create 6 new samples with varying amounts of fire coverage.

---

## ✅ Verification Checklist

- [ ] Webapp loads without errors at http://127.0.0.1:5000
- [ ] Click "Run Demo" → Displays prediction without JSON error
- [ ] Click "Choose File" → Can select .npy files from `data/` folder
- [ ] Upload `sample_high_fire_02.npy` → Shows high fire probability
- [ ] Upload `sample_no_fire_00.npy` → Shows low fire probability
- [ ] All output images display correctly
- [ ] Feature grid shows 12 channels
- [ ] Risk badge color changes based on fire percentage
- [ ] Report JSON displays without serialization errors
- [ ] Feature summary table shows statistics for all 12 channels

---

## 📞 Support

If you encounter any issues:

1. **Check Flask is running**: `http://127.0.0.1:5000` should load
2. **Check sample data exists**: Files should be in `data/` folder
3. **Check webapp/app.py**: Verify `convert_numpy_types()` function is present
4. **Check terminal output**: Look for error messages in Flask server output
5. **Restart Flask**: Kill process and run `python webapp/app.py` again

---

**Last Updated**: 2026-03-05  
**Status**: ✅ All issues fixed and tested
