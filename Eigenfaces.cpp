#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#define NR_FACES 12
#define K 6

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
double** getMatrix(uchar** faces_array, uchar* mean_array, int rows, int cols)
{
	double** matrix = new double* [rows];

	for (int i = 0; i < rows; i++)
		matrix[i] = new double[cols];

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
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

//functie pentru normalizarea matricei ce contine vectorii proprii
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

void swap(double* xp, double* yp)
{
	double temp = *xp;
	*xp = *yp;
	*yp = temp;
}

void selectionSort(double arr[])
{
	int i, j, max_idx;

	for (i = 0; i < NR_FACES - 1; i++)
	{
		max_idx = i;
		for (j = i + 1; j < NR_FACES; j++)
			if (arr[j] > arr[max_idx])
				max_idx = j;

		swap(&arr[max_idx], &arr[i]);
	}
}

int main()
{
	//citim fetele si le stocam intr-un tablou
	//https://github.com/j2kun/eigenfaces

	string path = "Images/person0/face0.jpg";
	Mat faces_img[NR_FACES];

	for (int i = 0; i < NR_FACES; i++) {
		if(i<=10)
			path.replace(13, 1, to_string(i));
		else
			path.replace(13, 2, to_string(i));
		faces_img[i] = imread(path, IMREAD_GRAYSCALE);
	}

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
	double** matrix = getMatrix(faces_array, mean_array, length, NR_FACES);

	//convertim tabloul bidimensional la o matrice din libraria Eigen
	MatrixXd eMatrix = ConvertToEigenMatrix(matrix, length, NR_FACES);

	//calculam matricea de covarianta
	MatrixXd covMatrix = eMatrix.transpose() * eMatrix;

	//calculam vectorii si valorile proprii
	EigenSolver<MatrixXd> s(covMatrix);

	MatrixXd D = s.pseudoEigenvalueMatrix();
	MatrixXd V = s.pseudoEigenvectors();
	//cout << "Matricea cu valorile proprii este:" << endl << D << endl;
	//cout << "Matricea cu vectorii proprii este:" << endl << V << endl;

	double* eigenvalues = new double[NR_FACES];
	double* eigenvalues_sorted = new double[NR_FACES];
	for (int i = 0; i < NR_FACES; i++) {
		eigenvalues[i] = D.coeff(i, i);
		eigenvalues_sorted[i] = D.coeff(i, i);
	}

	selectionSort(eigenvalues_sorted);

	for (int i = 0; i < NR_FACES; i++)
		for (int j = 0; j < NR_FACES; j++)
			if (eigenvalues_sorted[i] == eigenvalues[j])
				D.col(i) = V.col(j);

	//mapam vectorii proprii in matricea C' folosind relatia ui = A * vi
	MatrixXd U = eMatrix * D;

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
	Mat eigenfaces[NR_FACES];

	for (int i = 0; i < NR_FACES; i++)
		eigenfaces[i] = Mat(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[i]);

	//afisam eigenfaces
	hconcat(eigenfaces[0], eigenfaces[1], l0c0);
	hconcat(eigenfaces[2], eigenfaces[3], l0c1);
	hconcat(eigenfaces[4], eigenfaces[5], l1c0);
	hconcat(eigenfaces[6], eigenfaces[7], l1c1);
	hconcat(eigenfaces[8], eigenfaces[9], l2c0);
	hconcat(eigenfaces[10], eigenfaces[11], l2c1);

	hconcat(l0c0, l0c1, l1);
	hconcat(l1c0, l1c1, l2);
	hconcat(l2c0, l2c1, l3);

	vconcat(l1, l2, l1);
	vconcat(l1, l3, l1);

	imshow("eigenfaces", l1);

	//reprezentam fetele initiale prin combinatii liniare ale vectorilor proprii
	MatrixXd coeff_faces(NR_FACES, NR_FACES);
	for (int i = 0; i < NR_FACES; i++)
		coeff_faces.col(i) = U.colPivHouseholderQr().solve(eMatrix.col(i));

	//cout << "Matricea cu coeficienti:\n " << coeff_faces << endl;
	

	//citim o alta imagine cu aceeasi persoana si calculam coeficientii
	Mat input_face = imread("Images/person3/face2.jpg", IMREAD_GRAYSCALE);

	uchar** input_array = new uchar *;
	input_array[0] = input_face.isContinuous() ? input_face.data : input_face.clone().data;

	double** input_matrix = getMatrix(input_array, mean_array, length, 1);
	MatrixXd input_eMatrix = ConvertToEigenMatrix(input_matrix, length, 1);
	
	MatrixXd coeff_input(NR_FACES, 1);
	coeff_input.col(0) = U.colPivHouseholderQr().solve(input_eMatrix.col(0));

	//cout << "Vectorul de coeficienti pentru input:\n " << coeff_input << endl;

	//gasim distanta minima

	MatrixXf::Index min_index;
	(coeff_faces.colwise() - coeff_input.col(0)).colwise().squaredNorm().minCoeff(&min_index);

	//cout << coeff_faces.col(min_index) << endl;

	//imshow("input image", input_face);
	//imshow("person recognized", faces_img[min_index]);

	Mat final_result;
	hconcat(input_face, faces_img[min_index], final_result);
	imshow("input face + recognized person", final_result);

	waitKey();
	return 0;
}
