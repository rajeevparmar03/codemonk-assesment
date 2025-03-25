# codemonk-assesment

# Fashion Classification Model

##  Project Overview
This project is a multi-label classification model designed to predict multiple attributes of fashion items, such as **gender**,** article type**, base color, and season from an image. It uses a **pretrained EfficientNet-B0** model as a feature extractor and is fine-tuned for multi-label classification.

## 📂 Dataset
The dataset is sourced from a CSV file (`styles.csv`) containing fashion metadata and an image directory (`images/`). The relevant columns used are:
- **id** - Unique identifier for each image.
- **gender** - Gender category (e.g., Men, Women, Unisex).
- **articleType** - Type of clothing (e.g., T-shirt, Jeans, Shoes).
- **baseColour** - Primary color of the item.
- **season** - Suggested season for the fashion item (e.g., Summer, Winter).

## 🛠️ Model Architecture
- **Base Model**: `EfficientNet-B0` (pretrained on ImageNet)
- **Custom Fully Connected Layers**: Separate layers for each category prediction:
  - `fc_gender`
  - `fc_type`
  - `fc_color`
  - `fc_season`
- **Activation Function**: Softmax for multi-class classification
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (Learning Rate = 0.001)

## 🔄 Training Pipeline
### **1️⃣ Data Preprocessing**
- Loads the dataset and filters necessary columns.
- Applies **label encoding** to categorical attributes using `sklearn.preprocessing.LabelEncoder`.
- Uses `PIL` to load images and applies `torchvision.transforms` for preprocessing.
- Handles missing images with a custom `collate_fn` in the DataLoader.

### **2️⃣ Training Loop**
- Uses **PyTorch's DataLoader** for batch processing.
- Performs **forward pass**, calculates loss, and updates weights.
- Runs for multiple epochs, printing loss updates.

## ✅ How to Train the Model
```python
python train.py  # Runs the training script
```
This script:
- Initializes the dataset and DataLoader.
- Trains the model for the specified number of epochs.
- Saves the trained model (`fashion_model.pth`) and label encoders (`label_encoders.pkl`).

## 🚀 How to Test the Model
```python
python test.py --image_path "path/to/image.jpg"
```
This script:
- Loads the trained model and encoders.
- Applies image transformations.
- Runs inference and **decodes predictions** back to human-readable labels.

### **Example Output:**
```bash
Predicted Labels: {'gender': 'Men', 'articleType': 'T-Shirt', 'baseColour': 'Black', 'season': 'Summer'}
```

## 📝 Saving & Loading the Model Correctly
### **Saving Model and Encoders**
```python
import torch
import pickle

# Save model weights
torch.save(model.state_dict(), "fashion_model.pth")

# Save encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
```

### **Loading Model and Encoders**
```python
import torch
import pickle

# Load encoders
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load model
model = MultiLabelCNN(num_classes_list)
model.load_state_dict(torch.load("fashion_model.pth", map_location=device))
model.to(device)
model.eval()
```
```

