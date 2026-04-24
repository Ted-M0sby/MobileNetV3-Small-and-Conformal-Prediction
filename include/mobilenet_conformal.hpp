#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mobilenet {

struct PredictedClass {
    int classIndex = -1;
    std::string className;
    double probability = 0.0;
    double conformityScore = 0.0;
};

struct PredictionResult {
    PredictedClass basePrediction;
    std::vector<PredictedClass> predictionSet;
    std::size_t setSize = 0;
    double confidenceLevel = 0.95;
    bool uncertainty = false;
    std::string error;
};

class IImageClassifierBackend {
public:
    virtual ~IImageClassifierBackend() = default;
    virtual std::vector<double> predictProbabilities(const std::string& imagePath) const = 0;
    virtual std::size_t numClasses() const = 0;
};

class MockMobileNetBackend final : public IImageClassifierBackend {
public:
    explicit MockMobileNetBackend(std::size_t classes = 1000) : classes_(classes) {}

    std::vector<double> predictProbabilities(const std::string& imagePath) const override {
        std::mt19937_64 rng(hashString(imagePath));
        std::gamma_distribution<double> gamma(1.0, 1.0);

        std::vector<double> probs(classes_);
        double sum = 0.0;
        for (double& v : probs) {
            v = gamma(rng);
            sum += v;
        }

        if (sum == 0.0) {
            throw std::runtime_error("mock backend generated invalid distribution");
        }

        for (double& v : probs) {
            v /= sum;
        }

        return probs;
    }

    std::size_t numClasses() const override { return classes_; }

private:
    static std::uint64_t hashString(const std::string& s) {
        const std::uint64_t offset = 1469598103934665603ULL;
        const std::uint64_t prime = 1099511628211ULL;
        std::uint64_t h = offset;
        for (unsigned char c : s) {
            h ^= c;
            h *= prime;
        }
        return h;
    }

    std::size_t classes_;
};

class MobileNetConformal {
public:
    explicit MobileNetConformal(double confidenceLevel = 0.95,
                                IImageClassifierBackend* backend = nullptr)
        : confidenceLevel_(confidenceLevel),
          backendOwned_(backend == nullptr),
          backend_(backendOwned_ ? static_cast<IImageClassifierBackend*>(new MockMobileNetBackend()) : backend) {
        if (confidenceLevel_ <= 0.0 || confidenceLevel_ >= 1.0) {
            throw std::invalid_argument("confidenceLevel must be in (0, 1)");
        }
        if (backend_ == nullptr) {
            throw std::invalid_argument("backend cannot be null");
        }

        classNames_.reserve(backend_->numClasses());
        for (std::size_t i = 0; i < backend_->numClasses(); ++i) {
            classNames_.push_back("class_" + std::to_string(i));
        }
    }

    ~MobileNetConformal() {
        if (backendOwned_) {
            delete backend_;
        }
    }

    void calibrate(const std::vector<std::string>& imagePaths, const std::vector<int>& trueLabels) {
        if (imagePaths.size() != trueLabels.size() || imagePaths.empty()) {
            throw std::invalid_argument("imagePaths/trueLabels size mismatch or empty");
        }

        std::vector<double> scores;
        scores.reserve(imagePaths.size());

        for (std::size_t i = 0; i < imagePaths.size(); ++i) {
            const auto probs = backend_->predictProbabilities(imagePaths[i]);
            const int y = trueLabels[i];
            if (y < 0 || static_cast<std::size_t>(y) >= probs.size()) {
                throw std::out_of_range("label index out of range in calibration");
            }
            scores.push_back(1.0 - probs[static_cast<std::size_t>(y)]);
        }

        threshold_ = calculateThreshold(scores, confidenceLevel_);
        calibrated_ = true;
    }

    PredictionResult predict(const std::string& imagePath, int topK = 5) const {
        PredictionResult result;
        result.confidenceLevel = confidenceLevel_;

        if (!calibrated_) {
            result.error = "call calibrate() first";
            return result;
        }

        if (topK <= 0) {
            result.error = "topK must be positive";
            return result;
        }

        const auto probs = backend_->predictProbabilities(imagePath);
        const auto top = topKIndices(probs, static_cast<std::size_t>(topK));

        if (top.empty()) {
            result.error = "no prediction";
            return result;
        }

        const int bestIdx = top.front();
        result.basePrediction = makePredictedClass(bestIdx, probs[static_cast<std::size_t>(bestIdx)]);

        for (int idx : top) {
            const double p = probs[static_cast<std::size_t>(idx)];
            const double score = 1.0 - p;
            if (score <= threshold_) {
                result.predictionSet.push_back(makePredictedClass(idx, p));
            }
        }

        result.setSize = result.predictionSet.size();
        result.uncertainty = result.setSize > 1;
        return result;
    }

    std::vector<PredictionResult> batchPredict(const std::vector<std::string>& imagePaths, int topK = 5) const {
        std::vector<PredictionResult> out;
        out.reserve(imagePaths.size());
        for (const auto& p : imagePaths) {
            out.push_back(predict(p, topK));
        }
        return out;
    }

    std::unordered_map<std::string, std::string> getModelInfo() const {
        std::unordered_map<std::string, std::string> info;
        info["model_name"] = "MobileNetV3-Small (C++ rewrite, backend-pluggable)";
        info["num_classes"] = std::to_string(backend_->numClasses());
        info["calibrated"] = calibrated_ ? "true" : "false";
        info["confidence_level"] = toString(confidenceLevel_);
        info["backend"] = backendOwned_ ? "MockMobileNetBackend" : "Custom";
        return info;
    }

private:
    static std::string toString(double value) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << value;
        return oss.str();
    }

    PredictedClass makePredictedClass(int idx, double prob) const {
        PredictedClass c;
        c.classIndex = idx;
        c.className = classNames_[static_cast<std::size_t>(idx)];
        c.probability = prob;
        c.conformityScore = 1.0 - prob;
        return c;
    }

    static std::vector<int> topKIndices(const std::vector<double>& probs, std::size_t k) {
        std::vector<int> idx(probs.size());
        std::iota(idx.begin(), idx.end(), 0);

        if (k > idx.size()) {
            k = idx.size();
        }

        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                          [&probs](int a, int b) { return probs[static_cast<std::size_t>(a)] > probs[static_cast<std::size_t>(b)]; });
        idx.resize(k);
        return idx;
    }

    static double calculateThreshold(std::vector<double> scores, double confidenceLevel) {
        std::sort(scores.begin(), scores.end());
        const std::size_t n = scores.size();
        const double rank = std::ceil((static_cast<double>(n) + 1.0) * confidenceLevel);

        std::size_t idx = static_cast<std::size_t>(rank);
        if (idx == 0) {
            idx = 1;
        }
        if (idx > n) {
            idx = n;
        }

        return scores[idx - 1];
    }

    double confidenceLevel_;
    bool backendOwned_ = false;
    IImageClassifierBackend* backend_ = nullptr;
    std::vector<std::string> classNames_;
    double threshold_ = 0.0;
    bool calibrated_ = false;
};

} // namespace mobilenet
