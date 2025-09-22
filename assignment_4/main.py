import cv2
import numpy as np


def harris_corner_detection(reference_image):
    gray_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    reference_image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imwrite("solutions/harris.jpg", reference_image)


def feature_alignment(image_to_align, reference_image, max_features, good_match_percent):
    MIN_MATCH_COUNT = 10

    # Convert to grayscale
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_align, None)

    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform KNN matching
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = gray_ref.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        image_to_align = cv2.polylines(image_to_align, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv2.drawMatches(reference_image, kp1, image_to_align, kp2, good, None, **draw_params)
    cv2.imwrite("solutions/matches.jpg", img3)


def image_stitching(img1, img2, ratio=0.85, min_match=10, smoothing_window_size=800):
    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform KNN matching
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_points = []
    good_matches = []

    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m1, m2 = match_pair
            if m1.distance < ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

    # Draw matches
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
    #cv2.imwrite('solutions/matching.jpg', img3)

    if len(good_points) > min_match:
        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)

        # Create panorama
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        # Create masks for blending
        offset = int(smoothing_window_size / 2)
        barrier = img1.shape[1] - int(smoothing_window_size / 2)

        # Left image mask
        mask1 = np.zeros((height_panorama, width_panorama))
        mask1[:, barrier - offset:barrier + offset] = np.tile(
            np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
        mask1[:, :barrier - offset] = 1
        mask1 = cv2.merge([mask1, mask1, mask1])

        # Right image mask
        mask2 = np.zeros((height_panorama, width_panorama))
        mask2[:, barrier - offset:barrier + offset] = np.tile(
            np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
        mask2[:, barrier + offset:] = 1
        mask2 = cv2.merge([mask2, mask2, mask2])

        # Create panoramas
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1

        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2

        result = panorama1 + panorama2

        # Crop the result to remove black borders
        rows, cols = np.where(result[:, :, 0] != 0)
        if len(rows) > 0 and len(cols) > 0:
            min_row, max_row = min(rows), max(rows) + 1
            min_col, max_col = min(cols), max(cols) + 1
            final_result = result[min_row:max_row, min_col:max_col, :]

            cv2.imwrite('solutions/aligned.jpg', final_result)
            return final_result

    return None


if __name__ == "__main__":
    image1 = cv2.imread("../resources/reference_img.png")
    image2 = cv2.imread("../resources/align_this.jpg")

    # harris_corner_detection(image1)
    # feature_alignment(image2, image1, 10, 0.7)
    #image_stitching(image1, image2)
