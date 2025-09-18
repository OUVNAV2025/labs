// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  Fall 2025  - Lab 3 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to implement a BRISK feature tracker derived class
//  that inherits from your FeatureTracker base class.
//
// NOTE: Deliverables for the TEAM portion of this assignment start at number 4
// and end at number 11.  Deliverables 1-3 are
// individual.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  DELIVERABLE 7 (continued) | Comparing Feature Matching on Real Data
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
// For this part, you will need to implement the same functions you've just
// implemented in the case of SIFT, AKAZE, and ORB, but now for BRISK.
//
// NOTE: We do not provide much template code for this section or the
// corresponding header file (brisk_feature_tracker.h), but you should define
// and implement the same set of functions as for SIFT and AKAZE (see the
// template code for those cases for reference as needed)
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include "brisk_feature_tracker.h"

#include <glog/logging.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;

BriskFeatureTracker::BriskFeatureTracker()
    : FeatureTracker(), detector(BRISK::create()) {}

void BriskFeatureTracker::detectKeypoints(const cv::Mat& img,
                                          std::vector<KeyPoint>* keypoints) const {
  CHECK_NOTNULL(keypoints);
  // ~~~~ begin solution
  //
  //     **** TODO: FILL IN HERE ***
  //
  // ~~~~ end solution
}

void BriskFeatureTracker::describeKeypoints(const cv::Mat& img,
                                            std::vector<KeyPoint>* keypoints,
                                            cv::Mat* descriptors) const {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  // ~~~~ begin solution
  //
  //     **** TODO: FILL IN HERE ***
  //
  // ~~~~ end solution
}

void BriskFeatureTracker::matchDescriptors(
    const cv::Mat& descriptors_1,
    const cv::Mat& descriptors_2,
    std::vector<std::vector<DMatch>>* matches,
    std::vector<cv::DMatch>* good_matches) const {
  CHECK_NOTNULL(matches);

  // ~~~~ begin solution
  //
  //     **** TODO: FILL IN HERE ***
  //
  // ~~~~ end solution
}
