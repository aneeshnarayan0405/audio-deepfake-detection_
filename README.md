# Audio Deepfake Detection Assessment

## Part 1: Research & Selection

I reviewed the GitHub repository on audio deepfake detection:  
**[Audio-Deepfake-Detection](https://github.com/media-sec-lab/Audio-Deepfake-Detection)**  
Based on my analysis, I selected the following three approaches:

### **1. AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks**
- **Key Technical Innovation**:
  - Introduces spectro-temporal graph attention networks, integrating heterogeneous stacking graph attention layers.
  - Enhances feature extraction by modeling spectral and temporal patterns simultaneously.
  - Utilizes a stack node attention mechanism to improve spoofing artifact detection.
- **Reported Performance Metrics**:
  - Equal Error Rate (EER): 0.83% (ASVspoof 2019 LA dataset).
  - Tandem Detection Cost Function (t-DCF): 0.028.
- **Why This Approach?**
  - The graph-based attention mechanism makes it highly effective in detecting synthetic modifications
  - Strong generalization ability across different types of deepfake audio.
  - Well-suited for real-time detection due to its efficient feature extraction process.
- **Challenges**:
  - High computational cost for training, which may require specialized hardware.
  - Performance can degrade if applied to unseen deepfake types not represented in the training data.

### **2. End-to-End Anti-Spoofing with RawNet2**
- **Key Technical Innovation**:
  - Processes raw audio waveforms directly, eliminating the need for handcrafted feature extraction.
  - Leverages a CNN-based deep learning pipeline for automated feature learning.
  - Improves robustness to unseen spoofing techniques by focusing on end-to-end representation learning.
- **Reported Performance Metrics**:
  - Equal Error Rate (EER): 1.12%.
  - Tandem Detection Cost Function (t-DCF): 0.033.
- **Why This Approach?**
  - Lightweight architecture, making it ideal for real-time and low-latency applications.
  - Simplifies preprocessing by directly working with waveform data.
  - Generalizes well across different datasets, reducing overfitting to specific spoofing techniques.
- **Challenges**:
  - May require larger datasets for training to enhance performance on unseen deepfake techniques.
  - Direct waveform processing can lead to overfitting on limited training samples if not regularized properly.

### **3. One-Class Learning for Synthetic Voice Spoofing Detection**
- **Key Technical Innovation**:
  - Uses one-class learning with an OC-Softmax loss function to model bona fide speech distributions.
  - Detects spoofing by identifying deviations from the learned distribution of real human speech.
  - Focuses on anomaly detection, making it suitable for identifying unknown deepfake attacks.
- **Reported Performance Metrics**:
  - Equal Error Rate (EER): 2.19%.
  - Handles varying speech patterns.
- **Why This Approach?**
  - Can detect unknown spoofing attacks that were not present in the training set.
  - Works well for applications where deepfake patterns continuously evolve and new spoofing methods appear.
  - Lower computational requirements compared to fully supervised deep learning approaches.
- **Challenges**:
  - Performance is slightly lower than state-of-the-art fully supervised models.
  - Sensitivity to outliers: may produce false positives for natural speech variations not included in training.
 



## Part 2: Implementation

### **Selected Approach: CNN-Based Deepfake Detection**
- **Why?**:
  - Balanced trade-off between accuracy and computational efficiency.
  - Can be deployed in real-time applications.

### **Dataset Used**
- **WaveFake: DeepFake Audio Detection Dataset**:
  - Contains real and deepfake audio samples.
  - Used for fine-tuning our model.

### **Implementation Steps**
1. **Feature Extraction**:
   - Extract Mel-Frequency Cepstral Coefficients (MFCCs) from audio files.
   - Normalize and reshape features for CNN input.

2. **CNN Model Architecture**:
   - **Conv1D layers** for feature extraction.
   - **MaxPooling layers** for dimensionality reduction.
   - **Dense and Dropout layers** for classification.

3. **Training and Evaluation**:
   - Used **80% training / 20% testing** split.
   - Optimized using **Adam optimizer**.
   - Evaluated using **accuracy and F1-score**.

4. **Comparison with Other Approaches**:
   - Compared CNN with **Random Forest** and **LSTM-based models**.
   - CNN outperformed Random Forest and had similar accuracy to LSTM but was more computationally efficient.
  

## Dataset Used  
- **Name:** FakeAudio  
- **Source:** [FakeAudio Dataset on Kaggle](https://www.kaggle.com/datasets/walimuhammadahmad/fakeaudio)  
- **Description:** Contains real and AI-generated audio samples for deepfake detection.



## Part 3: Documentation & Analysis

### **Challenges Encountered**
- Handling dataset imbalance (solved via data augmentation).
- Optimizing hyperparameters for generalization.
- Computational constraints on training larger models.

### **Technical Explanation of CNN Model**
- **Feature Input**: Extracted spectrogram/MFCC features.
- **Convolutional Layers**: Capture patterns in frequency.
- **Pooling Layers**: Reduce dimensionality for efficiency.
- **Dense Layers**: Classify real vs. fake audio.

### **Performance Results**
- **Accuracy**: ~98% on WaveFake: DeepFake Audio Detection Dataset.
- **F1-Score**: ~0.94.
- **Inference Time**: ~50ms per sample.

### **Strengths and Weaknesses**
- **Strengths**:
  - Efficient real-time classification.
  - Robust against most deepfake manipulations.
- **Weaknesses**:
  - May fail against unseen deepfake methods.
  - Sensitive to noisy environments.

### **Future Improvements**
- Use **Transformer-based architectures (Wav2Vec 2.0)** for better feature extraction.
- Implement **adversarial training** to improve generalization.
- Optimize model for **edge-device deployment**.

### **Reflection Questions**
1. **Challenges Faced?**
   - Data imbalance, hyperparameter tuning, and generalization.
2. **Real-World Performance?**
   - Might underperform in highly noisy or unseen environments.
3. **Additional Data Needed?**
   - More diverse deepfake samples to improve robustness.
4. **Production Deployment?**
   - Use TensorFlow Serving or Flask API with lightweight model optimization.

