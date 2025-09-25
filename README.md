# MLP_for_CPP_Deployment

This repository provides an example of deploying an **MLP (Multi-Layer Perceptron)** model with **ONNX Runtime in C++**, using pre-scaled data.  
The implementation is designed to support both **regression** and **classification**, but currently it has only been tested for **regression**.

To train the MLP model for regression, you can follow this repository:  
👉 [MLP_for_Regression](https://github.com/Barelang63-KRSBI-Beroda/MLP_for_Regression.git)

When trained with that repo, the scaler files and model will be generated and can be directly used here.  
For testing purposes, this repository already includes a regression model trained on a Kaggle dataset:  
👉 [Student Performance Dataset](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set?resource=download)

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

In `main.cpp`, you can switch between **regression** and **classification** modes.  
Simply comment or uncomment the corresponding lines depending on the type of model you want to use:

```cpp
// ====== Mode Regression ======
float reg_result = mlp->PredictRegression(input_data);
std::cout << "Regression result: " << reg_result << std::endl;

// ====== Mode Classification ======
// int cls_result = mlp->PredictClassification(input_data);
// std::cout << "Classification result: class " << cls_result << std::endl;
```
