# MLP_for_CPP_Deployment

This repository provides an example of deploying an **MLP (Multi-Layer Perceptron)** model with **ONNX Runtime in C++**, using pre-scaled data.  
The implementation is designed to support both **regression** and **classification**.

To train and export an MLP model to ONNX format, use one of the Python repositories:      
👉 [MLP_for_Regression](https://github.com/Barelang63-KRSBI-Beroda/MLP_for_Regression.git)   
👉 [MLP_for_Classification](https://github.com/dsyahput/MLP_for_Classification.git)    
> When trained with those repositories, the ONNX model and scaler files will be generated and can be directly used here.

For testing purposes, this repository already includes sample models trained on Kaggle datasets:     
👉 [Student Performance Dataset](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set?resource=download)         
👉 [Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

---

## 🔧 Build Instructions

1. Install **ONNX Runtime (C++ API)**  
   You can build it or download prebuilt binaries from the official repo:  
   👉 [ONNX Runtime](https://github.com/microsoft/onnxruntime.git)

2. Update the path to your ONNX Runtime installation in `CMakeLists.txt`:  
   ```cmake
   set(ONNXRUNTIME_DIR "/path/to/onnxruntime")
   ```

3. Build the project:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. Run the executable:
   ```bash
   ./mlp_main
   ```

---

## ⚡ Usage Notes

In main.cpp, you can switch between Regression and Classification modes.
- Regression mode requires scaler_y.txt (target scaler).
- Classification mode ignores scaler_y.txt if provided

Simply comment or uncomment the corresponding block depending on the type of model you want to use:
```cpp
// ================= Regression Mode =================
mlp = new MLPModel(model_path, scaler_x_path, scaler_y_path, Mode::REGRESSION);
float reg_result = mlp->PredictRegression(input_data);
std::cout << "Regression result: " << reg_result << std::endl;

// ================= Classification Mode =================
mlp = new MLPModel(model_path, scaler_x_path, scaler_y_path, Mode::CLASSIFICATION);
int cls_result = mlp->PredictClassification(input_data);
std::cout << "Classification result: class " << cls_result << std::endl;
```
