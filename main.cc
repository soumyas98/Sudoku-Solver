#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <unistd.h>

using namespace cv::ml;
using namespace cv;
using namespace std;

string filename = "sudoku3.jpg";

// UNASSIGNED is used for empty cells in sudoku
#define UNASSIGNED 0
#define RED "\x1b[31m"
#define RESET "\x1b[0m"

// N is used for size of Sudoku grid. Size will be NxN
#define N 9

HOGDescriptor hog(
        Size(20, 20), //winSize
        Size(10, 10), //blocksize
        Size(5, 5),   //blockStride, 
        Size(10, 10), //cellSize, 
                 9, //nbins, 
                  1, //derivAper, 
                 -1, //winSigma, 
                  0, //histogramNormType, 
                0.2, //L2HysThresh, 
                  0, //gammal correction, 
                  64, //nlevels=64
                  1);

bool findUnassignedLocation(int grid[N][N], int &row, int &col);
void convertToMat(vector<vector<float>> &HOG, Mat &HOGMat);
void applyML(Mat&, Mat&);
void transformSVM(const Mat &response);
void prettyPrint(int grid[N][N]);
void templateMatching(const vector<Mat> &cells);
Mat process(Mat &cell);
bool taken[N * N];

// Checks whether it will be legal to assign num to the given row,col
bool isSafe(int grid[N][N], int row, int col, int num);
void getCellImgs(Mat &grid, vector<Mat> &cells);
void createHOG(vector<vector<float> > &HOG, vector<Mat> &cells);  

//method for solving
bool solveSudoku(int grid[N][N]) {
	int row, col;

	// If there is no unassigned location,sudoku is solved
	if (!findUnassignedLocation(grid, row, col)) {
		return true;
	}

	// numbers 1 to 9 will be tested
	for (int num = 1; num <= 9; num++) {
		if (isSafe(grid, row, col, num)) {
			// make first possible legal  assignment
			grid[row][col] = num;
			if (solveSudoku(grid)) {
				return true;
			}

			grid[row][col] = UNASSIGNED;
		}
	}

	// this triggers backtracking.If no number satisfies the square this means we went wrong and it starts going back.
	// Though a genuine problem is if some num is wrong somewhere it starts filling the peviously filled square bby starting off with 1 again.Think!!!
	return false; 
}

bool findUnassignedLocation(int grid[N][N], int &row, int &col) {
	for (row = 0; row < N; row++) {
		for (col = 0; col < N; col++) {
			if (grid[row][col] == UNASSIGNED) {
				return true;
			}
		}
	}

	return false;
}

//checks whether the entry is in some row or column
bool usedInRow(int grid[N][N], int row, int num) {
	for (int col = 0; col < N; col++) {
		if (grid[row][col] == num) {
			return true;
		}
	}

	return false;
}

bool usedInCol(int grid[N][N], int col, int num) {
	for (int row = 0; row < N; row++) {
		if (grid[row][col] == num) {
			return true;
		}
	}

	return false;
}

bool usedInBox(int grid[N][N], int boxStartRow, int boxStartCol, int num) {
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			if (grid[row+boxStartRow][col+boxStartCol] == num) {
				return true;
            }
        }
    }

	return false;
}

bool isSafe(int grid[N][N], int row, int col, int num) {
	/* Check if 'num' is not already placed in current row,
	current column and current 3x3 box */
	return !usedInRow(grid, row, num) &&
		!usedInCol(grid, col, num) &&
		!usedInBox(grid, row - row % 3 , col - col % 3, num);
}

void printGrid(int grid[N][N]) {
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			printf("%2d", grid[row][col]);
		}

		printf("\n");
	}
}

void showImage(String title, const Mat &img) {
	imshow(title, img);
    waitKey(0);
}

int main( int argc, char** argv ) {
	// Read original image 
	Mat src = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

	//if fail to read the image
	if (!src.data) {
		cout << "Error loading the image" << endl;
		return -1;
	}

	Mat srcb;
	cvtColor(src, srcb, COLOR_BGR2GRAY);

    showImage("Original image", src);

	Mat smooth;
	Mat thresholded;

    // removing noises, thresholding
	GaussianBlur(srcb, smooth, Size(7, 7), 0, 0); 
	adaptiveThreshold(smooth, thresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 5);

	//creating a copy
	Mat thresholded2 = thresholded.clone();   

    showImage("Smooth image", thresholded);

	vector<vector<Point>> contours; 
	vector<Vec4i> heirarchy;

	//FINDING CONTOUR
	findContours(thresholded2, contours, heirarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	//finding the sudoku with max area which will be our main grid
	double maxarea = 0;
	int p;
	for (int i = 0; i < contours.size(); i++) {
	    double area = contourArea(contours[i], false);
		if (area > 16 && area > maxarea) {
			maxarea = area;
			p = i;
		}
	}

	double perimeter = arcLength(contours[p], true);
	approxPolyDP(contours[p], contours[p], 0.01 * perimeter, true);
	drawContours(src, contours, p, Scalar(255, 0, 0), 3, 8);

	showImage("Contour image", src);

	Point2f in[4];
	Point2f out[4];

	int a, b, c, d; 
	double sum, prevsum, diff1, diff2, diffprev2, diffprev, prevsum2 = contours[p][0].x + contours[p][0].y;
	sum = prevsum = diffprev2 = diffprev = 0;

	for (int i = 0; i < 4; i++) {
		sum = contours[p][i].x + contours[p][i].y;
		diff1 = contours[p][i].x - contours[p][i].y;
		diff2 = contours[p][i].y - contours[p][i].x;
		if (diff1 > diffprev) {
			diffprev = diff1;
			c = i;
		}

		if (diff2 > diffprev2) {
			diffprev2 = diff2;
			d = i;
		}

		if (sum > prevsum) {
			prevsum = sum; 
			a = i;
		}

		if (sum < prevsum2) {
			prevsum2 = sum;
			b = i;
		}
	}

	in[0] = contours[p][a];
	in[1] = contours[p][b];
	in[2] = contours[p][c];
	in[3] = contours[p][d];

	out[0] = Point2f(450, 450);
	out[1] = Point2f(0, 0);
	out[2] = Point2f(450, 0);
	out[3] = Point(0, 450);

	Mat wrap, mat;
	mat = Mat::zeros(src.size(), src.type());
	wrap = getPerspectiveTransform(in, out);

	warpPerspective(src, mat, wrap, Size(450, 450));

    showImage("Sudoku part", mat);

	Mat ch, thresholded31;
	cvtColor(mat, ch, CV_BGR2GRAY);
	GaussianBlur(ch, ch, Size(11, 11), 0, 0);
	adaptiveThreshold(ch, thresholded31, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 7, 2);

	int p2 = 0, p3 = 0;
	while (p3 < 450) {
		for (int i = p3; i < p3 + 10; i++) {
			for (int j = 0; j < 450; j++) {
				thresholded31.at<uchar>(j,i) = 0;
			}
		}
		p3 = p3 + 50;
	}

	while (p2 < 450) {
		for (int i = 0; i < 450; i++) {
			for (int j = p2; j < p2 + 10; j++) {
				thresholded31.at<uchar>(j, i) = 0;
			}
		}
		p2 = p2 + 50;
	}

    showImage("Thresholded New", thresholded31);

    // stores each cell as 20 x 20 image after processing.
    vector<Mat> cells;
    getCellImgs(thresholded31, cells);
    
    // generate feature vector using histogram of oriented graphics.
    vector<vector<float>> HOG;
    createHOG(HOG, cells);

    // convert vetor to cv::Mat
    Mat HOGMat(HOG.size(), HOG[0].size(), CV_32FC1);
    convertToMat(HOG, HOGMat);

    // apply svm and get response, which will contain 81 digits.
    Mat response;
    applyML(response, HOGMat);

    transformSVM(response);
}
    
void transformSVM(const Mat &response) {
    int grid[N][N];
    for (int i = 0, j = 0, k = 0; i < response.rows; ++i, ++j) {
        grid[k][j] = response.at<float>(i, 0);

        if (j != 0 && j % 8 == 0) {
            ++k;
            j = -1;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (grid[i][j] != 0) {
                taken[i * 9 + j] = true;
            }
        }
    }

    if (solveSudoku(grid)) {
        cout << "Solved\n";
    } else {
        cout << "Cannot be solved\n";
    }

    prettyPrint(grid);
}

void prettyPrint(int grid[N][N]) {
    for (int i = 0; i < 9; i++) {
        if (i % 3 == 0) {
            printf(" +-----------+-----------+----------+\n");
        }

        for (int j = 0; j < 9; j++) {
            if (j % 3 == 0) {
                printf(" | ");
            }

            if (taken[i * 9 + j]) {
                printf(RED " %d " RESET, grid[i][j]);
            } else {
                printf(" %d ", grid[i][j]);
            }


            if (j == 8) {
                printf("|");
            }
        }

        printf("\n");
    }

    printf(" +-----------+-----------+----------+\n");
}


Mat process(Mat &cell) {
	vector<vector<Point>> contours; 
	vector<Vec4i> heirarchy;

	findContours(cell, contours, heirarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    int largestArea = 0, idx = -1;
    for (int i = 0; i < contours.size(); ++i) {
        double a = contourArea(contours[i], false);
        if (a > largestArea) {
            largestArea = a;
            idx = i;
        }
    }

	if (idx == -1 || largestArea < 120) {
	    return Mat::zeros(20, 20, cell.type());
	}
	
	Rect rect = boundingRect(contours[idx]);
	Mat roi = cell(rect).clone();
	
	Mat paddedRoi;
	int top = 0.25 * roi.rows;
	int bottom = top;
	int left = 0.25 * roi.cols;
	int right = left;
	copyMakeBorder(roi, paddedRoi, top, bottom, left, right, BORDER_CONSTANT, Scalar(0, 0, 0));
	
	Mat resizedCell = Mat(20, 20, cell.type());
	resize(paddedRoi, resizedCell, Size(20, 20));
	
	return resizedCell;
}

void applyML(Mat &response, Mat &HOGMat) {
    Ptr<SVM> svm = Algorithm::load<SVM>("digits-classification/model4.yml");
    svm->predict(HOGMat, response);
}

void createHOG(vector<vector<float> > &HOG, vector<Mat> &cells) {
    for (Mat &cell : cells) {
        vector<float> descriptors;
    	hog.compute(cell, descriptors);
    	HOG.push_back(descriptors);
    }
}

void convertToMat(vector<vector<float>> &HOG, Mat &HOGMat) {
    for (int i = 0; i < HOG.size(); ++i) {
        for (int j = 0; j < HOG[0].size(); ++j) {
            HOGMat.at<float>(i, j) = HOG[i][j];
        }
    }
}

void getCellImgs(Mat &grid, vector<Mat> &cells) {
    for (int i = 0; i <= 400; i += 50) {
        for (int j = 0; j <= 400; j += 50) {
            Mat cell = grid(Range(i, i + 50), Range(j, j + 50)); 
            Mat processedCell = process(cell);
            cells.push_back(processedCell);
        }
    }
}
