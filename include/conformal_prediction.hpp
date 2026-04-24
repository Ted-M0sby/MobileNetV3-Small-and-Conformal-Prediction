#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace conformal {

class ConformalPredictor {
public:
    explicit ConformalPredictor(double confidenceLevel = 0.95)
        : confidenceLevel_(confidenceLevel) {
        if (confidenceLevel_ <= 0.0 || confidenceLevel_ >= 1.0) {
            throw std::invalid_argument("confidenceLevel must be in (0, 1)");
        }
    }

    virtual ~ConformalPredictor() = default;

    double confidenceLevel() const { return confidenceLevel_; }
    double threshold() const {
        if (!fitted_) {
            throw std::logic_error("predictor is not calibrated");
        }
        return threshold_;
    }

protected:
    void setCalibrationScores(std::vector<double> scores) {
        if (scores.empty()) {
            throw std::invalid_argument("calibration scores cannot be empty");
        }
        calibrationScores_ = std::move(scores);
        threshold_ = calculateCalibrationThreshold(calibrationScores_, confidenceLevel_);
        fitted_ = true;
    }

    static double calculateCalibrationThreshold(const std::vector<double>& scores, double confidenceLevel) {
        std::vector<double> sorted = scores;
        std::sort(sorted.begin(), sorted.end());

        const std::size_t n = sorted.size();
        const double rank = std::ceil((static_cast<double>(n) + 1.0) * confidenceLevel);
        std::size_t idx = static_cast<std::size_t>(rank);
        if (idx == 0) {
            idx = 1;
        }
        if (idx > n) {
            idx = n;
        }

        return sorted[idx - 1];
    }

    bool isFitted() const { return fitted_; }

private:
    double confidenceLevel_;
    std::vector<double> calibrationScores_;
    double threshold_ = 0.0;
    bool fitted_ = false;
};

class ConformalRegressor : public ConformalPredictor {
public:
    using PredictorFn = std::function<double(const std::vector<double>&)>;

    explicit ConformalRegressor(double confidenceLevel = 0.95)
        : ConformalPredictor(confidenceLevel) {}

    void calibrate(const std::vector<std::vector<double>>& xCalib,
                   const std::vector<double>& yCalib,
                   PredictorFn predictor) {
        if (xCalib.size() != yCalib.size() || xCalib.empty()) {
            throw std::invalid_argument("xCalib/yCalib size mismatch or empty");
        }

        predictor_ = std::move(predictor);
        std::vector<double> scores;
        scores.reserve(xCalib.size());

        for (std::size_t i = 0; i < xCalib.size(); ++i) {
            const double pred = predictor_(xCalib[i]);
            scores.push_back(std::abs(pred - yCalib[i]));
        }

        setCalibrationScores(std::move(scores));
    }

    std::vector<std::pair<double, double>> predictIntervals(const std::vector<std::vector<double>>& xTest) const {
        if (!isFitted()) {
            throw std::logic_error("regressor is not calibrated");
        }

        std::vector<std::pair<double, double>> intervals;
        intervals.reserve(xTest.size());

        for (const auto& x : xTest) {
            const double pred = predictor_(x);
            intervals.emplace_back(pred - threshold(), pred + threshold());
        }

        return intervals;
    }

private:
    PredictorFn predictor_;
};

class ConformalClassifier : public ConformalPredictor {
public:
    using ProbFn = std::function<std::vector<double>(const std::vector<double>&)>;

    explicit ConformalClassifier(double confidenceLevel = 0.95)
        : ConformalPredictor(confidenceLevel) {}

    void calibrate(const std::vector<std::vector<double>>& xCalib,
                   const std::vector<int>& yCalib,
                   ProbFn probFn) {
        if (xCalib.size() != yCalib.size() || xCalib.empty()) {
            throw std::invalid_argument("xCalib/yCalib size mismatch or empty");
        }

        probFn_ = std::move(probFn);
        std::vector<double> scores;
        scores.reserve(xCalib.size());

        for (std::size_t i = 0; i < xCalib.size(); ++i) {
            const auto probs = probFn_(xCalib[i]);
            const int y = yCalib[i];
            if (y < 0 || static_cast<std::size_t>(y) >= probs.size()) {
                throw std::out_of_range("label index out of range");
            }
            scores.push_back(1.0 - probs[static_cast<std::size_t>(y)]);
        }

        setCalibrationScores(std::move(scores));
    }

    std::vector<int> predictSet(const std::vector<double>& x, const std::vector<int>& candidateLabels) const {
        if (!isFitted()) {
            throw std::logic_error("classifier is not calibrated");
        }

        const auto probs = probFn_(x);
        std::vector<int> set;

        for (int label : candidateLabels) {
            if (label < 0 || static_cast<std::size_t>(label) >= probs.size()) {
                continue;
            }
            const double score = 1.0 - probs[static_cast<std::size_t>(label)];
            if (score <= threshold()) {
                set.push_back(label);
            }
        }

        return set;
    }

private:
    ProbFn probFn_;
};

} // namespace conformal
