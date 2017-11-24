#include <image2images/File.h>
#include <iostream>

std::shared_ptr<Image> File::getImage(int index) {
    if(index < 0 || index >= m_images.size()) {
        return nullptr;
    }
    return m_images[index];
}

bool File::read(std::string fileName, unsigned int numOfLine) {
    std::cout << ">> Loading images ..." << std::endl;
    std::ifstream inputFile(fileName);
    if(!inputFile.is_open()) {
        return false;
    }

    // Get the number of matrices in file
    unsigned int totalMats;
    inputFile >> totalMats;

    // Update total mats if specified by the user
    if(numOfLine > 0) {
        totalMats = std::min(totalMats, numOfLine);
    }

    // Resize vector
    m_images.resize(totalMats);

    // Start reading matrices
    for(int line=0; line < totalMats; line++) {

        // Get label
        int label;
        inputFile >> label;

        // Get side
        int side;
        inputFile >> side;

        // Create matrix
        std::shared_ptr<cv::Mat> matrix = std::make_shared<cv::Mat>(side, side, CV_8UC1);

        // Populate matrix
        for(int row=0; row < side; row++) {
            for(int col=0; col < side; col++) {
                double pixelVal;
                inputFile >> pixelVal;
                matrix->at<uchar>(row, col) = (uchar) pixelVal;
            }
        }

        // Store image
        m_images[line] = std::make_shared<Image>(label, matrix);

        // Log
        if(line % 1000 == 0) {
            std::cout << ">> Loaded " << line+1 << " out of " << totalMats << std::endl;
        }
    }
    return true;
}