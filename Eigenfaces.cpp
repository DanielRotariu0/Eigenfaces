#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#define NR_FACES 12

using namespace cv;
using namespace std;
using namespace Eigen;

//functie pentru calculul fetei medii
uchar* getMeanArray(uchar** faces_array, int length)
{
	uchar* mean_array = new uchar[length];

	for (int i = 0; i < length; i++) {
		int sum = 0;

		for (int j = 0; j < NR_FACES; j++)
			sum += faces_array[j][i];

		mean_array[i] = sum / NR_FACES;
	}

	return mean_array;
}

//functie pentru calculul matricei ce contine fetele - fata medie
double** getMatrix(uchar** faces_array, uchar* mean_array, int length)
{
	double** matrix = new double* [length];

	for (int i = 0; i < length; i++)
		matrix[i] = new double[NR_FACES];

	for (int i = 0; i < length; i++)
		for (int j = 0; j < NR_FACES; j++)
			matrix[i][j] = faces_array[j][i] - mean_array[i];

	return matrix;
}

//functie pentru a converti un tablou bidimensional la o matrice din libraria Eigen
MatrixXd ConvertToEigenMatrix(double** data, int rows, int cols)
{
	MatrixXd eMatrix(rows, cols);

	for (int i = 0; i < rows; ++i)
		eMatrix.row(i) = VectorXd::Map(&data[i][0], cols);

	return eMatrix;
}

//functie pentru aflarea minimului dintr-un vector din libraria Eigen
double minEigenVector(VectorXd v, int length)
{
	double min = DBL_MAX;

	for (int i = 0; i < length; i++)
		if (v[i] < min)
			min = v[i];

	return min;
}

//functie pentru aflarea maximului dintr-un vector din libraria Eigen
double maxEigenVector(VectorXd v, int length)
{
	double max = DBL_MIN;

	for (int i = 0; i < length; i++)
		if (v[i] > max)
			max = v[i];

	return max;
}

//functie pentru normalizarea matricei ce contine vectorii proproo
MatrixXd normalize(MatrixXd V2, int length, int val)
{
	for (int i = 0; i < NR_FACES; i++) {

		double min = minEigenVector(V2.col(i), length);
		double max = maxEigenVector(V2.col(i), length);

		for (int j = 0; j < length; j++) {

			V2.col(i)[j] = (V2.col(i)[j] - min) / (max - min) * val;

			if (V2.col(i)[j] < 0)
				V2.col(i)[j] = 0;

			if (V2.col(i)[j] > 255)
				V2.col(i)[j] = 255;
		}
	}

	return V2;
}

int main()
{
	//citim fetele si le stocam intr-un tablou
	Mat faces_img[NR_FACES];
	faces_img[0]  = imread("Images/face0.png", IMREAD_GRAYSCALE);
	faces_img[1]  = imread("Images/face1.png", IMREAD_GRAYSCALE);
	faces_img[2]  = imread("Images/face2.png", IMREAD_GRAYSCALE);
	faces_img[3]  = imread("Images/face3.png", IMREAD_GRAYSCALE);
	faces_img[4]  = imread("Images/face4.png", IMREAD_GRAYSCALE);
	faces_img[5]  = imread("Images/face5.png", IMREAD_GRAYSCALE);
	faces_img[6]  = imread("Images/face6.png", IMREAD_GRAYSCALE);
	faces_img[7]  = imread("Images/face7.png", IMREAD_GRAYSCALE);
	faces_img[8]  = imread("Images/face8.png", IMREAD_GRAYSCALE);
	faces_img[9]  = imread("Images/face9.png", IMREAD_GRAYSCALE);
	faces_img[10] = imread("Images/face10.png", IMREAD_GRAYSCALE);
	faces_img[11] = imread("Images/face11.png", IMREAD_GRAYSCALE);

	//verificam daca s-au citit corect imaginile
	for (int i = 0; i < NR_FACES; i++)
		if (faces_img[i].empty()) {
			cout << "Image File Not Found: " << i << endl;
			cin.get();
			return -1;
		}

	//afisam imaginile citite
	Mat l0c0, l0c1, l1c0, l1c1, l2c0, l2c1;
	Mat l1, l2, l3;

	hconcat(faces_img[0], faces_img[1], l0c0);
	hconcat(faces_img[2], faces_img[3], l0c1);
	hconcat(faces_img[4], faces_img[5], l1c0);
	hconcat(faces_img[6], faces_img[7], l1c1);
	hconcat(faces_img[8], faces_img[9], l2c0);
	hconcat(faces_img[10], faces_img[11], l2c1);

	hconcat(l0c0, l0c1, l1);
	hconcat(l1c0, l1c1, l2);
	hconcat(l2c0, l2c1, l3);

	vconcat(l1, l2, l1);
	vconcat(l1, l3, l1);

	imshow("faces", l1);

	//reprezentam fetele sub forma unei matrice
	uchar** faces_array = new uchar * [NR_FACES];
	uint length = faces_img[0].total() * faces_img[0].channels();

	for (int i = 0; i < NR_FACES; i++)
		faces_array[i] = faces_img[i].isContinuous() ? faces_img[i].data : faces_img[i].clone().data;

	//calculam fata medie
	uchar* mean_array = getMeanArray(faces_array, length);

	//convertim vectorul la un obiect de tip Mat
	Mat mean_face(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), mean_array);
	imshow("mean face", mean_face);

	//calculam matricea ce contine fetele - fata medie
	double** matrix = getMatrix(faces_array, mean_array, length);

	//convertim tabloul bidimensional la o matrice din libraria Eigen
	MatrixXd eMatrix = ConvertToEigenMatrix(matrix, length, NR_FACES);

	//calculam matricea de covarianta
	MatrixXd covMatrix = eMatrix.transpose() * eMatrix;

	//calculam vectorii si valorile proprii
	EigenSolver<MatrixXd> s(covMatrix);

	MatrixXd D = s.pseudoEigenvalueMatrix();
	MatrixXd V = s.pseudoEigenvectors();
	cout << "Matricea cu valorile proprii este:" << endl << D << endl;
	cout << "Matricea cu vectorii proprii este:" << endl << V << endl;

	//mapam vectorii proprii in matricea C' folosind relatia ui = A * vi
	MatrixXd U = eMatrix * V;

	//normalizam vectorii proprii
	U = normalize(U, length, 255);

	//extragem vectorii proprii
	uchar** eigenvectors = new uchar * [NR_FACES];

	for (int i = 0; i < NR_FACES; i++)
		eigenvectors[i] = new uchar[length];

	for (int i = 0; i < NR_FACES; i++)
		for (int j = 0; j < length; j++)
			eigenvectors[i][j] = U.col(i)[j];

	//reprezentam vectorii proprii sub forma unor obiecte de tip Mat
	Mat eigenface0(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[0]);
	Mat eigenface1(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[1]);
	Mat eigenface2(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[2]);
	Mat eigenface3(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[3]);
	Mat eigenface4(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[4]);
	Mat eigenface5(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[5]);
	Mat eigenface6(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[6]);
	Mat eigenface7(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[7]);
	Mat eigenface8(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[8]);
	Mat eigenface9(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[9]);
	Mat eigenface10(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[10]);
	Mat eigenface11(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[11]);

	hconcat(eigenface0, eigenface1, l0c0);
	hconcat(eigenface2, eigenface3, l0c1);
	hconcat(eigenface4, eigenface5, l1c0);
	hconcat(eigenface6, eigenface7, l1c1);
	hconcat(eigenface8, eigenface9, l2c0);
	hconcat(eigenface10, eigenface11, l2c1);

	hconcat(l0c0, l0c1, l1);
	hconcat(l1c0, l1c1, l2);
	hconcat(l2c0, l2c1, l3);

	vconcat(l1, l2, l1);
	vconcat(l1, l3, l1);

	imshow("eigenfaces", l1);

	waitKey();
	return 0;
}
