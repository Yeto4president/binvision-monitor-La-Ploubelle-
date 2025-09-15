# 🗑️ Smart Bin Monitoring Platform - Wild Dump Prevention (WDP)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.x-green?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.x-orange?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

**Intelligent image-based bin monitoring platform for preventing illegal dumping**

</div>

## 📋 Project Overview

Wild Dump Prevention (WDP) is an innovative web platform that uses image analysis to monitor public bin fill levels, helping municipalities prevent overflow and illegal dumping through proactive waste management.

## 🎯 Key Features

### 📸 Image Management
- **Upload System**: Citizen/agent image submission with format validation (JPG, PNG)
- **Manual Annotation**: Interactive interface for bin state classification (Full/Empty)
- **Automatic Processing**: Background feature extraction upon upload

### 🔍 Visual Feature Extraction
- **Basic Metrics**: File size, dimensions, average RGB values
- **Advanced Analysis**: Color histograms, contrast levels, edge detection
- **Metadata Storage**: All features stored in SQLite database

### 🤖 Smart Classification
- **Rule-Based AI**: Conditional classification without heavy machine learning
- **Customizable Rules**: Configurable thresholds for automatic bin state detection
- **Transparent Logic**: Easily auditable and modifiable decision system

### 📊 Interactive Dashboard
- **Real-time Statistics**: Fill rate distribution, temporal trends, location mapping
- **Dynamic Visualizations**: Chart.js integration for interactive graphs
- **Risk Mapping**: Geographic visualization of potential overflow zones

## 🛠️ Technical Stack

### Backend
- **Framework**: Flask (Python)
- **Image Processing**: OpenCV, Pillow
- **Database**: SQLite
- **Data Analysis**: Pandas, NumPy

### Frontend
- **UI Framework**: HTML5, CSS3, Bootstrap
- **Visualization**: Chart.js, Matplotlib
- **Interactivity**: JavaScript, jQuery

# 🔍 Classification Rules

```python
def classify_bin(image_features):
    """
    Rule-based classification for bin fill status
    Uses simple conditional logic based on image characteristics
    """
    if (image_features['avg_color'] < 100 and 
        image_features['file_size'] > 500000):
        return "Full"
    elif (image_features['contrast'] > 50 and 
          image_features['edges_count'] < 1000):
        return "Empty"
    return "Unknown"
```
### 📊 Dashboard Metrics
- Total Images: Upload count statistics
- Fill Distribution: Full vs Empty percentage
- Temporal Analysis: Upload trends over time
- Geographic Data: Location-based risk mapping

### 🌱 Green IT Implementation
#### Environmental Considerations
- ♻️ Low Resource Usage: Minimal computational requirements
- ⚡ Energy Efficiency: No heavy ML training needed
- 🌍 Sustainable Design: Local processing reduces cloud dependency
- 💻 Hardware Longevity: Compatible with older systems

### 📈 Performance Metrics
- Metric	Value
- Image Processing Time	< 2 seconds
- Classification Accuracy	~85% (rule-based)
- Maximum Concurrent Users	50+
- Database Response Time	< 100 ms
  
### 👥 Contributors
#### Core Team
- Tristan Garnier 
- Armand Colonna 
- Raphaël Soave 
- Antonin Gabet 
- Louis Lazauche 
- Jonathan Ayeto 
