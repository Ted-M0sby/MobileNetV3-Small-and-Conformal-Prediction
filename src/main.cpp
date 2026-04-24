#include <iostream>
#include <string>
#include <vector>

#include "mobilenet_conformal.hpp"

int main() {
    try {
        mobilenet::MobileNetConformal classifier(0.95);

        std::vector<std::string> calibImages = {
            "calib_img1.jpg", "calib_img2.jpg", "calib_img3.jpg", "calib_img4.jpg", "calib_img5.jpg"
        };
        std::vector<int> calibLabels = {0, 1, 2, 3, 4};

        classifier.calibrate(calibImages, calibLabels);

        auto result = classifier.predict("test_image.jpg", 5);
        if (!result.error.empty()) {
            std::cout << "Predict error: " << result.error << '\n';
            return 1;
        }

        std::cout << "Base prediction: "
                  << result.basePrediction.className
                  << " (idx=" << result.basePrediction.classIndex
                  << ", p=" << result.basePrediction.probability << ")\n";
        std::cout << "Prediction set size: " << result.setSize << '\n';
        std::cout << "Uncertainty: " << (result.uncertainty ? "true" : "false") << '\n';

        for (const auto& item : result.predictionSet) {
            std::cout << "  - " << item.className
                      << " (idx=" << item.classIndex
                      << ", p=" << item.probability
                      << ", score=" << item.conformityScore << ")\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal: " << ex.what() << '\n';
        return 1;
    }
}
