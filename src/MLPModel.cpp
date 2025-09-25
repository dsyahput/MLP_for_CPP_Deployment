#include "MLPModel.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

MLPModel::MLPModel(const std::string& model_path, 
                   const std::string& scaler_x_path,
                   const std::string& scaler_y_path) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "ONNX_NN"), session_options_(), session_(nullptr)
{
    session_options_.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    mean_x_ = load_scaler_row(scaler_x_path, 0);
    std_x_  = load_scaler_row(scaler_x_path, 1);

    auto mean_y_vec = load_scaler_row(scaler_y_path, 0);
    auto std_y_vec  = load_scaler_row(scaler_y_path, 1);

    if (mean_y_vec.empty() || std_y_vec.empty()) {
        throw std::runtime_error("Scaler_y file is invalid or empty.");
    }

    mean_y_ = mean_y_vec[0];
    std_y_  = std_y_vec[0];
}

std::vector<float> MLPModel::load_scaler_row(const std::string& filename, int row) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open scaler file: " + filename);
    }

    std::string line;
    for (int i = 0; i <= row; ++i) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Row " + std::to_string(row) + " not found in " + filename);
        }
    }

    std::istringstream iss(line);
    std::vector<float> values;
    std::string token;
    while (std::getline(iss, token, ',')) {
        values.push_back(std::stof(token));
    }

    if (values.empty()) {
        throw std::runtime_error("Row " + std::to_string(row) + " is empty in " + filename);
    }

    return values;
}

std::vector<float> MLPModel::scale_input(const std::vector<float>& input) {
    if (input.size() != mean_x_.size() || input.size() != std_x_.size()) {
        throw std::runtime_error("Input size does not match scaler_x size.");
    }

    std::vector<float> scaled;
    scaled.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        scaled.push_back((input[i] - mean_x_[i]) / std_x_[i]);
    }
    return scaled;
}

float MLPModel::unscale_output(float scaled_y) {
    return scaled_y * std_y_ + mean_y_;
}

// ============================= Regression ==============================
float MLPModel::PredictRegression(const std::vector<float>& input_x) {
    std::vector<float> input_scaled = scale_input(input_x);
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_scaled.size())};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_scaled.data(), input_scaled.size(),
        input_shape.data(), input_shape.size());

    auto input_name = session_->GetInputNameAllocated(0, allocator_);
    auto output_name = session_->GetOutputNameAllocated(0, allocator_);

    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                       input_names.data(), &input_tensor, 1,
                                       output_names.data(), 1);

    float y_scaled = output_tensors.front().GetTensorMutableData<float>()[0];
    return unscale_output(y_scaled);
}

// ============================= Classification ==============================
int MLPModel::PredictClassification(const std::vector<float>& input_x) {
    std::vector<float> input_scaled = scale_input(input_x);
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_scaled.size())};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_scaled.data(), input_scaled.size(),
        input_shape.data(), input_shape.size());

    auto input_name = session_->GetInputNameAllocated(0, allocator_);
    auto output_name = session_->GetOutputNameAllocated(0, allocator_);

    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                       input_names.data(), &input_tensor, 1,
                                       output_names.data(), 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    auto shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    if (shape.size() < 2) {
        throw std::runtime_error("Expected classification output with shape [1, num_classes]");
    }

    size_t num_classes = static_cast<size_t>(shape[1]);

    // Argmax
    int predicted_class = 0;
    float max_val = output_data[0];
    for (size_t i = 1; i < num_classes; ++i) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            predicted_class = static_cast<int>(i);
        }
    }

    return predicted_class;
}
