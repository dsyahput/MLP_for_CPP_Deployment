#include <iostream>
#include <vector>
#include <string>
#include "MLPModel.h"

int main() {
    // ===== Path variables =====
    std::string model_path    = "../models/regression/best_model/model.onnx";
    std::string scaler_x_path = "../models/regression/scaler_X.txt";
    std::string scaler_y_path = "../models/regression/scaler_y.txt";

    MLPModel* mlp = nullptr;
    std::vector<float> input_data = {1,49,1,9,4};

    // ================= Mode Regression =================
    mlp = new MLPModel(model_path, scaler_x_path, scaler_y_path, Mode::REGRESSION);
    float reg_result = mlp->PredictRegression(input_data);
    std::cout << "Regression result: " << reg_result << std::endl;

    // ================= Mode Classification =================
    // mlp = new MLPModel(model_path, scaler_x_path, scaler_y_path, Mode::CLASSIFICATION);
    // int cls_result = mlp->PredictClassification(input_data);
    // std::cout << "Classification result: class " << cls_result << std::endl;
    
    delete mlp;


    return 0;
}
