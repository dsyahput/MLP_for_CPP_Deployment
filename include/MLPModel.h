#pragma once
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

enum class Mode {
    REGRESSION,
    CLASSIFICATION
};

class MLPModel
{
public:
    MLPModel(const std::string& model_path,
             const std::string& scaler_x_path,
             const std::string& scaler_y_path,
             Mode mode);

    // Prediction functions
    float PredictRegression(const std::vector<float>& input_data);
    int PredictClassification(const std::vector<float>& input_data);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    Mode mode_;

    std::vector<float> mean_x_;
    std::vector<float> std_x_;
    float mean_y_ = 0.0f;
    float std_y_  = 1.0f;

    // Helpers
    std::vector<float> load_scaler_row(const std::string& filename, int row);
    std::vector<float> scale_input(const std::vector<float>& input);
    float unscale_output(float scaled_y);
};
