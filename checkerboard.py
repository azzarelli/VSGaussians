import cv2
import numpy as np

# Load your checkerboard image
img = cv2.imread("checkerboard.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the checkerboard pattern (number of inner corners per row/col)
checkerboard_dims = (7, 6)  # adjust to your pattern (cols, rows)
square_size = 7.0  # mm, adjust to your real checker square size

# Prepare 3D object points (0,0,0), (1,0,0), ... in checkerboard plane
objp = np.zeros((checkerboard_dims[0]*checkerboard_dims[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
objp *= square_size  # scale to mm (or any consistent unit)

# Find corners in the image
ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

if ret:
    # Refine corner positions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # Run calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners2], gray.shape[::-1], None, None
    )

    print("Camera matrix K:\n", K)
    print(f"f_x = {K[0,0]:.2f}, f_y = {K[1,1]:.2f}")
    print(f"Principal point = ({K[0,2]:.2f}, {K[1,2]:.2f})")
else:
    print("Checkerboard not detected.")
