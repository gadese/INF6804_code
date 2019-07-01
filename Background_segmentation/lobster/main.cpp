#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lib_lobster/BackgroundSubtractorLOBSTER.h"

const std::string& DATASET_LOCATION = "/home/beaupreda/litiv/datasets/CD2014";

const std::string& DYNAMIC_BACKGROUND = "/dynamic_background";
const std::string& BASELINE = "/baseline";
const std::string& CAMERA_JITTERING = "/camera_jittering";
const std::string& NIGHT = "/night";

const std::string& FILE_DYN_BACK = "../dyn_back.txt";
const std::string& FILE_BASELINE = "../baseline.txt";
const std::string& FILE_CAM_JITTER = "../cam_jitter.txt";
const std::string& FILE_NIGHT = "../night.txt";

const std::string& DYNAMIC_BACKGROUND_LOCATION = DATASET_LOCATION + DYNAMIC_BACKGROUND;
const std::string& BASELINE_LOCATION = DATASET_LOCATION + BASELINE;
const std::string& CAMERA_JITTERING_LOCATION = DATASET_LOCATION + CAMERA_JITTERING;
const std::string& NIGHT_LOCATION = DATASET_LOCATION + NIGHT;

const std::string& OUTPUT_PATH = "../results";
const std::string& INPUT = "/input";
const std::string& TEMPORAL_ROI = "/temporalROI.txt";
const std::string& ROI = "/ROI.bmp";
const std::string& LOCAL_FOLDER = "../";


// thanks internet!
int getDirectoryInfo(const std::string& directory, std::vector<std::string>& files) {
    DIR* dp;
    struct dirent *dirp;
    if ((dp = opendir(directory.c_str())) == NULL) {
        return -1;
    }
    while ((dirp = readdir(dp)) != NULL) {
        // eliminate hidden directories...
        if (std::string(dirp->d_name)[0] != '.')
            files.push_back(std::string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

void getTemporalROI(const std::string& filename, unsigned long& first, unsigned long& last) {
    std::ifstream file(filename);
    file >> first;
    file >> last;
}

void createDirectories() {
    if (!std::system(("mkdir " + OUTPUT_PATH).c_str()));
        std::cout << "Folder " + OUTPUT_PATH + " created" << std::endl;
    if (!std::system(("mkdir " + OUTPUT_PATH + BASELINE).c_str()))
        std::cout << "Folder " + OUTPUT_PATH + BASELINE + " created!" << std::endl;
    if (!std::system(("mkdir " + OUTPUT_PATH + DYNAMIC_BACKGROUND).c_str()))
        std::cout << "Folder " + OUTPUT_PATH + DYNAMIC_BACKGROUND + " created!" << std::endl;
    if (!std::system(("mkdir " + OUTPUT_PATH + NIGHT).c_str()))
        std::cout << "Folder " + OUTPUT_PATH + NIGHT + " created!" << std::endl;
    if (!std::system(("mkdir " + OUTPUT_PATH + CAMERA_JITTERING).c_str()))
        std::cout << "Folder " + OUTPUT_PATH + CAMERA_JITTERING + " created!" << std::endl;
}

void saveBoxes(unsigned long frame, const std::vector<cv::Rect>& bboxes, std::ofstream& file) {
    for (int i = 0; i < bboxes.size(); i++) {
        file << frame << " " << bboxes[i].x << " " << bboxes[i].y << " " << bboxes[i].width << " " << bboxes[i].height << std::endl;
    }
}

void drawBoxes(cv::Mat img, const std::vector<cv::Rect>& bboxes) {
    cv::Mat img_color(img.size(), CV_8UC3);
    cv::cvtColor(img.clone(), img_color, CV_GRAY2BGR);
    for (int i = 0; i < bboxes.size(); i++) {
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::rectangle(img_color, bboxes[i].tl(), bboxes[i].br(), color, 1, 8, 0);
    }
    cv::imshow("Boxes", img_color);
    cv::waitKey(0);
}

int main() {
    createDirectories();

    const std::vector<std::string> locations = {BASELINE_LOCATION, NIGHT_LOCATION, CAMERA_JITTERING_LOCATION, DYNAMIC_BACKGROUND_LOCATION};
    const std::vector<std::string> names = {BASELINE, NIGHT, CAMERA_JITTERING, DYNAMIC_BACKGROUND};
    const std::vector<std::string> filenames = {FILE_BASELINE, FILE_NIGHT, FILE_CAM_JITTER, FILE_DYN_BACK};

    for (int i = 0; i < locations.size(); i++) {
        std::ofstream file(filenames[i]);
        std::vector<std::string> baselineFilenames;
        if ((getDirectoryInfo(locations[i] + INPUT, baselineFilenames) != 0)) {
            std::cerr << "Problem with locating input of the " << names[i] << " sequence" << std::endl;
            return -1;
        }

        sort(baselineFilenames.begin(), baselineFilenames.end(),
             [](std::string a, std::string b) {
                 return a < b;
             });

        unsigned long first = 0;
        unsigned long last = baselineFilenames.size();
        getTemporalROI(locations[i] + TEMPORAL_ROI, first, last);

        if (baselineFilenames.begin() + first - 1 > baselineFilenames.begin())
            baselineFilenames.erase(baselineFilenames.begin(), baselineFilenames.begin() + first - 1);
        if (baselineFilenames.begin() + last - 1 < baselineFilenames.end())
            baselineFilenames.erase(baselineFilenames.begin() + last - 1, baselineFilenames.end());

        cv::Mat currInputFrame, currSegmentMaskLobster;
        currInputFrame = cv::imread(locations[i] + INPUT + "/" + baselineFilenames[0], CV_LOAD_IMAGE_UNCHANGED);
        currSegmentMaskLobster.create(currInputFrame.size(), CV_8UC1);
        cv::Mat sequenceROI = cv::imread(locations[i] + ROI, CV_LOAD_IMAGE_UNCHANGED);

        BackgroundSubtractorLOBSTER lobster;
        lobster.initialize(currInputFrame, sequenceROI);

        for (int j = 0; j < baselineFilenames.size(); j++) {
            currInputFrame = cv::imread(locations[i] + INPUT + "/" + baselineFilenames[j], CV_LOAD_IMAGE_UNCHANGED);
            lobster(currInputFrame, currSegmentMaskLobster);
            imwrite(OUTPUT_PATH + names[i] + "/" + baselineFilenames[j], currSegmentMaskLobster);
            std::cout << j + 1 << " / " << baselineFilenames.size() << " images processed" << std::endl;

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(currSegmentMaskLobster.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
            std::vector<cv::Rect> bboxes(contours.size());
            for (int k = 0; k < contours.size(); k++) {
                bboxes[k] = cv::boundingRect(cv::Mat(contours[k]));
            }
            saveBoxes(first + j, bboxes, file);
            //drawBoxes(currSegmentMaskLobster.clone(), bboxes);
        }
    }
    return 0;
}