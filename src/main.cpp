#include <iostream>
#include <vector>
#include <string>
#include "MLPModel.h"

int main() {
    try {
        std::string model_path   = "../models/model_1/best_model/model.onnx";
        std::string scaler_x_path = "../models/model_1/scaler_X.txt";
        std::string scaler_y_path = "../models/model_1/scaler_y.txt";

        MLPModel* mlp = nullptr;
        mlp = new MLPModel(model_path, scaler_x_path, scaler_y_path);

        std::vector<float> input_data = {1,49,1,9,4};

        // ====== Mode Regression ======
        float reg_result = mlp->PredictRegression(input_data);
        std::cout << "Regression result: " << reg_result << std::endl;

        // ====== Mode Classification ======
        // int cls_result = mlp->PredictClassification(input_data);
        // std::cout << "Classification result: class " << cls_result << std::endl;
        
        delete mlp;
    }
    catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
