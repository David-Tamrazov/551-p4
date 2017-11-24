#include <image2images/Image.h>
#include <iostream>

#define NUM_OBJECTS 3

// Init unique id val
long Image::m_uniq_id = 1;

void Image::display() {
    std::stringstream winName;
    winName << "Image " << m_name << " # " << m_id;
    cv::namedWindow(winName.str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(winName.str(), *m_mat);
}

int Image::_reduceColors(int k) {
    int n = m_mat->rows * m_mat->cols;
    cv::Mat data = m_mat->reshape(1, n);
    data.convertTo(data, CV_32F);

    std::vector<int> labels;
    cv::Mat1f colors;
    cv::kmeans(data, k, labels
            , cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001)
            , 5, cv::KMEANS_PP_CENTERS, colors);

    int maxVal = 0;
    for (int i = 0; i < n; ++i) {
        data.at<float>(i, 0) = colors(labels[i], 0);
        maxVal = std::max(maxVal, (int)data.at<float>(i, 0));
    }

    cv::Mat reduced = data.reshape(1, m_mat->rows);
    reduced.convertTo(reduced, CV_8U);
    reduced.copyTo(*m_mat);
    return maxVal;
}

double Image::_averagePixelVal() {
    double sum = 0;
    for(int row=0; row < getSide(); row++) {
        for(int col=0; col < getSide(); col++) {
            sum += (int) m_mat->at<uchar>(row, col);
        }
    }
    return sum / (getSide()* getSide());
}

std::shared_ptr<Image> Image::binarize() {
    std::shared_ptr<Image> binImage = _cloneImage();

    // Enlarge
    int scale = 5;
    cv::resize(*binImage->getMat(), *binImage->getMat(),
               cv::Size(scale * binImage->getMat()->rows, scale * binImage->getMat()->cols));

    // Reduce colors
    int maxVal = binImage->_reduceColors(5);

    // Apply binary threshold
    cv::threshold(*binImage->getMat(), *binImage->getMat(), maxVal-1, 255, CV_THRESH_BINARY);

    // Dilate
    int dilateVal = 4;
    cv::Mat kernel(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateVal, dilateVal)));
    cv::dilate(*binImage->getMat(), *binImage->getMat(), kernel, cv::Point(-1, -1), 1);

    // Recover original size
    cv::resize(*binImage->getMat(), *binImage->getMat(),
               cv::Size(binImage->getMat()->rows/scale, binImage->getMat()->cols/scale));
    return binImage;
}

std::shared_ptr<Image> Image::align(int k) {
    std::vector<int> indices;
    std::vector<std::vector<cv::Point>> charsContour = _groupContours(k);
    for(int i=0; i < charsContour.size(); i++) {
        indices.push_back(i);
    }
    return _align(indices, charsContour);
}

std::shared_ptr<Image> Image::cleanNoise() {

    // Find contours
    std::vector<std::vector<cv::Point>> charsContour = _charsControus();
    std::shared_ptr<Image> cleanImage = _cloneImage();

    // Clean small elements
    for (size_t i=0; i< charsContour.size(); i++) {
        double area = cv::contourArea(charsContour[i], false);
        if(area < 20) {
            cv::drawContours(*cleanImage->getMat(), charsContour, (int)i, 0, CV_FILLED);
        }
    }

    return cleanImage;
}

std::shared_ptr<Image> Image::drawContour(int k) {

    // Clone current image
    std::shared_ptr<Image> contourImage = _cloneImage();

    // Find contours
    std::vector<std::vector<cv::Point>> charsContrours = contourImage->_groupContours(k);

    // Loop on each object
    for (size_t i=0; i<charsContrours.size(); i++) {

        // Store points
        std::vector<cv::Point> points;
        for (size_t j = 0; j < charsContrours[i].size(); j++) {
            cv::Point p = charsContrours[i][j];
            points.push_back(p);
        }

        // Draw rectangle
        if(points.size() > 0){
            cv::Rect brect = cv::boundingRect(cv::Mat(points).reshape(2));
            cv::rectangle(*contourImage->getMat(), brect.tl() - cv::Point(2,2), brect.br()+cv::Point(2,2),
                          cv::Scalar(220, 100, 200), 1, CV_AA);
        }
    }
    return contourImage;
}

std::shared_ptr<Image> Image::_align(const std::vector<int> &indices,
                                     std::vector<std::vector<cv::Point>> &charsContours) {

    // Compute new sizes
    int padding = 3;
    int widthSum = 0;
    int height = 0;
    for(int index=0; index < indices.size(); index++) {
        cv::Rect brect = cv::boundingRect(cv::Mat(charsContours[indices[index]]).reshape(2));
        widthSum += std::max(brect.width, brect.height) + padding;
        height = std::max(height, brect.height);
    }

    // Add border padding
    widthSum += 2*padding;
    height += 2*padding;

    // Determine new side value
    int side = std::max(widthSum, height);

    // Construct and initialize a new mat
    std::shared_ptr<Image> image = _cloneImage();
    image->m_mat = std::make_shared<cv::Mat>(side, side, m_mat->type());
    image->getMat()->setTo(cv::Scalar(0));

    // Start populating the new matrix
    int leftOffset = padding;
    for(size_t i=0; i < indices.size(); i++) {

        // Get object
        cv::Rect brect = cv::boundingRect(cv::Mat(charsContours[indices[i]]).reshape(2));
        cv::Mat elementMat = ((*m_mat)(cv::Rect(brect.tl(), brect.br())));

        // Create mask
        cv::Mat bigMask = cv::Mat::zeros(m_mat->rows, m_mat->cols, m_mat->type());
        drawContours(bigMask, charsContours, indices[i], cv::Scalar(255), CV_FILLED);
        cv::Mat smallMask = cv::Mat::zeros(image->getMat()->rows, image->getMat()->cols, image->getMat()->type());
        bigMask(cv::Rect(brect.x, brect.y, brect.width, brect.height)).copyTo(smallMask);

        // Compute top offset
        int topOffset = padding;
        if(side > brect.height) {
            topOffset = (side - brect.height) / 2;
        }

        // Compute left char offset
        int leftCharOffset = 0;
        if(brect.height > brect.width) {
            leftCharOffset = (brect.height - brect.width) / 2;
        }

        // Draw element on new matrix
        elementMat.copyTo(
                (*image->getMat())(cv::Rect(leftOffset + leftCharOffset, topOffset, brect.width, brect.height)), smallMask);

        // Update left offset
        leftOffset += std::max(brect.width, brect.height) + padding;
    }
    return image;
}

void Image::_permutation(std::vector<std::shared_ptr<Image>> &outputImages, std::vector<int> &indices,
                         std::vector<std::vector<cv::Point>> &charsContours) {
    if(indices.size() == charsContours.size()) {
        std::shared_ptr<Image> newImage = _align(indices, charsContours);
        if(newImage) {
            outputImages.push_back(newImage);
        }
    } else {
        for(int i=0; i < charsContours.size(); i++) {
            if(std::find(indices.begin(), indices.end(), i) == indices.end()) {
                indices.push_back(i);
                _permutation(outputImages, indices, charsContours);
                indices.pop_back();
            }
        }
    }
}

std::vector<std::shared_ptr<Image>> Image::permutation() {
    std::vector<std::shared_ptr<Image>> perImages;
    std::vector<int> indices;
    std::vector<std::vector<cv::Point>> charsContours = _groupContours(NUM_OBJECTS);
    _permutation(perImages, indices, charsContours);
    return perImages;
}

std::vector<std::shared_ptr<Image>> Image::split() {
    std::vector<std::shared_ptr<Image>> splitImages;

    // Group contours
    std::vector<std::vector<cv::Point>> charsContours = _groupContours(NUM_OBJECTS);
    for(int i=0; i < charsContours.size(); i++) {
        std::vector<int> indices = {i};
        std::shared_ptr<Image> elementImage = _align(indices, charsContours);
        if(elementImage) {
            splitImages.push_back(elementImage);
        }
    }
    return splitImages;
}

void Image::wait() {
    cv::waitKey(0);
}

std::shared_ptr<Image> Image::rotate(int angle) {
    std::shared_ptr<Image> rotImage = _cloneImage();
    cv::Point2f center(getSide()/2.0f, getSide()/2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Rect bbox = cv::RotatedRect(center,m_mat->size(), angle).boundingRect();
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;
    cv::warpAffine(*m_mat, *rotImage->getMat(), rot, bbox.size());
    return rotImage;
}

std::vector<std::shared_ptr<Image>> Image::mnist() {
    // Prepare mnist vector
    std::vector<std::shared_ptr<Image>> mnistVector;

     // Rotate images
    std::shared_ptr<Image> LRImage1 = rotate(45);
    std::shared_ptr<Image> LRImage2 = rotate(30);
    std::shared_ptr<Image> LRImage3 = rotate(20);
    std::shared_ptr<Image> LRImage4 = rotate(10);
    std::shared_ptr<Image> RRImage4 = rotate(-10);
    std::shared_ptr<Image> RRImage3 = rotate(-20);
    std::shared_ptr<Image> RRImage2 = rotate(-30);
    std::shared_ptr<Image> RRImage1 = rotate(-45);

     // Add new images
    mnistVector.push_back(LRImage1);
    mnistVector.push_back(LRImage2);
    mnistVector.push_back(LRImage3);
    mnistVector.push_back(LRImage4);
    mnistVector.push_back(shared_from_this());
    mnistVector.push_back(RRImage4);
    mnistVector.push_back(RRImage3);
    mnistVector.push_back(RRImage2);
    mnistVector.push_back(RRImage1);
    return mnistVector;
}

std::shared_ptr<Image> Image::_cloneImage() {
    return std::make_shared<Image>(m_label, _cloneMat());
}

std::shared_ptr<Image> Image::size(int side) {
    std::shared_ptr<Image> image = _cloneImage();
    cv::resize(*image->getMat(), *image->getMat(), cv::Size(side, side));
    return image;
}

std::shared_ptr<cv::Mat> Image::_cloneMat() {
    std::shared_ptr<cv::Mat> mat = std::make_shared<cv::Mat>(getSide(), getSide(), m_mat->type());
    m_mat->copyTo(*mat);
    return mat;
}

std::vector<std::vector<cv::Point>> Image::_charsControus() {
    std::vector<std::vector<cv::Point>> charsContours;
    cv::findContours(*m_mat, charsContours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    return charsContours;
}

std::vector<std::vector<cv::Point>> Image::_groupContours(int k) {
    std::vector<std::vector<cv::Point>> charsContour = _charsControus();

    // Check if already has k contours
    if(charsContour.size() == k) {
        return charsContour;
    }

    // If has less then k, then apply k means
    if(charsContour.size() < k) {

        // Get all non-zero pixels
        std::vector<cv::Point2f> points;
        for(int row=0; row < m_mat->rows; row++) {
            for(int col=0; col < m_mat->cols; col++) {
                if(m_mat->at<uchar>(row, col) != 0) {
                    points.push_back(cv::Point2f(col, row));
                }
            }
        }

        // Prepare the contour vector
        std::vector<std::vector<cv::Point>> kContours(k);

        // Apply k means
        cv::Mat labels;
        cv::Mat centers;
        cv::kmeans(points, k, labels,
                   cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 50, 1.0), 3,
                   cv::KMEANS_PP_CENTERS, centers);
        for(int i=0; i < points.size(); i++) {
            kContours[labels.at<uchar>(i, 0)].push_back(points[i]);
        }
        return kContours;
    }

    // If has more than k, then
    // compute center of each contour
    // and group the close ones until
    // there are exactly k groups
    std::vector<cv::Point> centers;
    for(int i=0; i < charsContour.size(); i++) {
        cv::Moments mmts = cv::moments(charsContour[i]);
        int x = (int) std::round(mmts.m10 / mmts.m00);
        int y = (int) std::round(mmts.m01 / mmts.m00);
        centers.push_back(cv::Point(x, y));
    }

    // Match the closest two
    std::map<int, std::vector<int>> groups;
    for(int repeat=0; repeat < charsContour.size() - k; repeat++) {
        int from = -1;
        int to = -1;
        for(int a=0; a < charsContour.size(); a++) {
            for(int b=a+1; b < charsContour.size(); b++) {
                // Check not added already
                double eDist = cv::norm(centers[a] - centers[b]);
                if(from == -1 || eDist < cv::norm(centers[from] - centers[to])) {
                    if(groups.find(a) == groups.end()
                       || std::find(groups[a].begin(), groups[a].end(), b) == groups[a].end()) {
                        from = a;
                        to = b;
                    }
                }
            }
        }

        if(from != -1) {
            groups[from].push_back(to);
        }
    }

    // Prepare contours
    std::vector<std::vector<cv::Point>> groupedContours;
    std::vector<bool> visited(charsContour.size(), false);

    // Add groups first
    for(auto group : groups) {
        if(!visited[group.first]) {
            std::queue<int> indexQueue;
            indexQueue.push(group.first);
            std::vector<cv::Point> contour;
            while(!indexQueue.empty()) {
                int poll = indexQueue.front();
                indexQueue.pop();
                visited[poll] = true;
                contour.insert(contour.end(), charsContour[poll].begin(), charsContour[poll].end());
                if(groups.find(poll) != groups.end()) {
                    for(int child : groups[poll]) {
                        if(!visited[child]) {
                            indexQueue.push(child);
                        }
                    }
                }
            }
            groupedContours.push_back(contour);
        }
    }

    // Add non-grouped contours
    for(int i=0; i < charsContour.size(); i++) {
        if(!visited[i]) {
            groupedContours.push_back(charsContour[i]);
        }
    }
    return groupedContours;
}

std::shared_ptr<Image> Image::blur() {
    std::shared_ptr<Image> blurImage = _cloneImage();
    cv::blur(*blurImage->getMat(), *blurImage->getMat(), cv::Size(2,2));
    return blurImage;
}