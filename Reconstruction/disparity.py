import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt 

def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

#Downsamples image x number (reduce_factor) of times. Assumes both images are the same size.  
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

ret = np.load('./camera_params/ret.npy')
K = np.load('./camera_params/K.npy')
dist = np.load('./camera_params/dist.npy')


#=========================================================
# Structure from motion (3D scene reconstruction)
#=========================================================

img_path1 = './reconstruct_this/left2.jpg'
img_path2 = './reconstruct_this/right2.jpg'


img_1 = cv2.imread(img_path1)
img_2 = cv2.imread(img_path2)


img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

h,w = img_2_gray.shape

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

#Undistort images

img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)

#Downsample each image 3 times (because they're too big)
img_1_downsampled = downsample_image(img_1_undistorted,3)
img_2_downsampled = downsample_image(img_2_undistorted,3)

cv2.imwrite('undistorted_left.jpg', img_1_downsampled)
cv2.imwrite('undistorted_right.jpg', img_2_downsampled)

#To do: fine tune disparity map. 

#disparity range is tuned according to specific parameters obtained in blog online. 

win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16

stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
	numDisparities = num_disp,
	blockSize = 5,
	uniquenessRatio = 5,
	speckleWindowSize = 5,
	speckleRange = 5,
	disp12MaxDiff = 2,
	P1 = 8*3*win_size**2,#8*3*win_size**2,
	P2 =32*3*win_size**2) #32*3*win_size**2)

print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

#Show disparity map before generating 3D cloud. Otherwise its not going to work. 

plt.imshow(disparity_map,'gray')
plt.show()



print ("\nGenerating the 3D map...")
h,w = img_2_downsampled.shape[:2]
#focal_length = np.load('./camera_params/FocalLength.npy')
focal_length = np.load('./camera_params/FocalLength.npy')

print(focal_length)
print(0.8*w)
#Perspective transformation matrix
Q = np.float32([[1,0,0,-w/2.0],
				[0,-1,0,h/2.0],
				[0,0,0,-focal_length],
				[0,0,1,0]])

Q2 = np.float32([[1,0,0,0],
				[0,-1,0,0],
				[0,0,focal_length*0.05,0],
				[0,0,0,1]])
points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)
mask_map = disparity_map > disparity_map.min()
output_points = points_3D[mask_map]
output_colors = colors[mask_map]
output_file = 'reconstructed.ply'

print ("\n Creating the output file... \n")
create_output(output_points, output_colors, output_file)

