#pragma once

#include <fstream>
#include <memory>
#include <image2images/Image.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class File {
private:
    std::vector<std::shared_ptr<Image>> m_images;
public:
    /**
     * Read file
     * @param fileName
     * @param numOfLine
     * @return true if successful
     */
    bool read(std::string fileName, unsigned int numOfLine);

    /**
     * Get total number of matrices
     * @return size
     */
    int getSize() const { return m_images.size();}

    /**
     * Load image
     * @return matrix pointer
     */
    std::shared_ptr<Image> getImage(int index);
};