#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#define NR_FACES 12
#define K 11

using namespace cv;
using namespace std;
using namespace Eigen;

// functie pentru calculul fetei medii
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

// functie pentru calculul matricei ce contine fetele - fata medie
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

// functie pentru a converti un tablou bidimensional la o matrice din libraria Eigen
MatrixXd ConvertToEigenMatrix(double** data, int rows, int cols)
{
	MatrixXd eMatrix(rows, cols);

	for (int i = 0; i < rows; ++i)
		eMatrix.row(i) = VectorXd::Map(&data[i][0], cols);

	return eMatrix;
}

// functie pentru aflarea minimului dintr-un vector din libraria Eigen
double minEigenVector(VectorXd v, int length)
{
	double min = DBL_MAX;

	for (int i = 0; i < length; i++)
		if (v[i] < min)
			min = v[i];

	return min;
}

// functie pentru aflarea maximului dintr-un vector din libraria Eigen
double maxEigenVector(VectorXd v, int length)
{
	double max = DBL_MIN;

	for (int i = 0; i < length; i++)
		if (v[i] > max)
			max = v[i];

	return max;
}

// functie pentru normalizarea matricei ce contine vectorii proprii
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

// functie pentru afisarea mai multor imagini in aceeasi fereastra
// https://github.com/opencv/opencv/wiki/DisplayManyImages
void ShowManyImages(string title, int nArgs, ...) {
	int size;
	int i;
	int m, n;
	int x, y;
	int w, h;

	float scale;
	int max;

	if (nArgs <= 0) {
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nArgs > 14) {
		printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
		return;
	}
	else if (nArgs == 1) {
		w = h = 1;
		size = 300;
	}
	else if (nArgs == 2) {
		w = 2; h = 1;
		size = 300;
	}
	else if (nArgs == 3 || nArgs == 4) {
		w = 2; h = 2;
		size = 300;
	}
	else if (nArgs == 5 || nArgs == 6) {
		w = 3; h = 2;
		size = 200;
	}
	else if (nArgs == 7 || nArgs == 8) {
		w = 4; h = 2;
		size = 200;
	}
	else {
		w = 4; h = 3;
		size = 150;
	}

	Mat DispImage = Mat::zeros(Size(100 + size * w, 60 + size * h), CV_8UC3);

	va_list args;
	va_start(args, nArgs);

	for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
		Mat img = va_arg(args, Mat);
		if (img.empty()) {
			printf("Invalid arguments");
			return;
		}
		x = img.cols;
		y = img.rows;
		max = (x > y) ? x : y;
		scale = (float)((float)max / size);

		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		Rect ROI(m, n, (int)(x / scale), (int)(y / scale));
		Mat temp; resize(img, temp, Size(ROI.width, ROI.height));
		temp.copyTo(DispImage(ROI));
	}

	namedWindow(title, 1);
	imshow(title, DispImage);

	va_end(args);
}

int main()
{
	// imaginile folosite au fost obtinute din acest repository:
	// https://github.com/j2kun/eigenfaces

	// citim fetele si le stocam intr-un tablou
	string path = "Images/person0/face0.jpg";
	Mat faces_img[NR_FACES];

	for (int i = 0; i < NR_FACES; i++) {
		if(i<=10)
			path.replace(13, 1, to_string(i));
		else
			path.replace(13, 2, to_string(i));
		faces_img[i] = imread(path, IMREAD_GRAYSCALE);
	}

	// verificam daca s-au citit corect imaginile
	for (int i = 0; i < NR_FACES; i++)
		if (faces_img[i].empty()) {
			cout << "Image File Not Found: " << i << endl;
			cin.get();
			return -1;
		}

	// afisam imaginile citite

	Mat faces_color[NR_FACES];
	for (int i = 0; i < NR_FACES; i++)
		cvtColor(faces_img[i], faces_color[i], COLOR_GRAY2RGB);

	ShowManyImages("Faces", 12, faces_color[0], faces_color[1], faces_color[2], faces_color[3],
		faces_color[4], faces_color[5], faces_color[6], faces_color[7], faces_color[8],
		faces_color[9], faces_color[10], faces_color[11]);

	// reprezentam fetele sub forma unei matrice
	uchar** faces_array = new uchar * [NR_FACES];
	uint length = faces_img[0].total() * faces_img[0].channels();

	for (int i = 0; i < NR_FACES; i++)
		faces_array[i] = faces_img[i].isContinuous() ? faces_img[i].data : faces_img[i].clone().data;

	// calculam fata medie
	uchar* mean_array = getMeanArray(faces_array, length);

	// convertim vectorul la un obiect de tip Mat
	Mat mean_face(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), mean_array);

	// afisam fata medie
	Mat mean_face_color;
	cvtColor(mean_face, mean_face_color, COLOR_GRAY2RGB);
	ShowManyImages("Mean Face", 1, mean_face_color);

	// calculam matricea ce contine fetele - fata medie
	double** matrix = getMatrix(faces_array, mean_array, length, NR_FACES);

	// convertim tabloul bidimensional la o matrice din libraria Eigen
	MatrixXd eMatrix = ConvertToEigenMatrix(matrix, length, NR_FACES);

	// calculam matricea de covarianta
	MatrixXd covMatrix = eMatrix.transpose() * eMatrix;

	// calculam vectorii si valorile proprii
	EigenSolver<MatrixXd> s(covMatrix);

	MatrixXd D = s.pseudoEigenvalueMatrix();
	MatrixXd V = s.pseudoEigenvectors();
	// cout << "Matricea cu valorile proprii este:" << endl << D << endl;
	// cout << "Matricea cu vectorii proprii este:" << endl << V << endl;

	// sortam vectorii proprii in functie de valorile proprii
	double* eigenvalues = new double[NR_FACES];
	double* eigenvalues_sorted = new double[NR_FACES];
	for (int i = 0; i < NR_FACES; i++) {
		eigenvalues[i] = D.coeff(i, i);
		eigenvalues_sorted[i] = D.coeff(i, i);
	}

	selectionSort(eigenvalues_sorted);

	// vectorii proprii se vor gasi in matricea D
	for (int i = 0; i < NR_FACES; i++)
		for (int j = 0; j < NR_FACES; j++)
			if (eigenvalues_sorted[i] == eigenvalues[j])
				D.col(i) = V.col(j);

	// mapam vectorii proprii in matricea C' folosind relatia ui = A * vi
	MatrixXd U = eMatrix * D;

	// normalizam vectorii proprii
	U = normalize(U, length, 255);

	// extragem vectorii proprii
	uchar** eigenvectors = new uchar * [NR_FACES];

	for (int i = 0; i < NR_FACES; i++)
		eigenvectors[i] = new uchar[length];

	for (int i = 0; i < NR_FACES; i++)
		for (int j = 0; j < length; j++)
			eigenvectors[i][j] = U.col(i)[j];

	// reprezentam vectorii proprii sub forma unor obiecte de tip Mat pentru a putea fi afisati
	Mat eigenfaces[NR_FACES];

	for (int i = 0; i < NR_FACES; i++)
		eigenfaces[i] = Mat(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[i]);

	// afisam eigenfaces
	Mat eigenfaces_color[NR_FACES];
	for (int i = 0; i < NR_FACES; i++)
		cvtColor(eigenfaces[i], eigenfaces_color[i], COLOR_GRAY2RGB);

	ShowManyImages("Eigenfaces", 12, eigenfaces_color[0], eigenfaces_color[1], eigenfaces_color[2], eigenfaces_color[3],
		eigenfaces_color[4], eigenfaces_color[5], eigenfaces_color[6], eigenfaces_color[7], eigenfaces_color[8],
		eigenfaces_color[9], eigenfaces_color[10], eigenfaces_color[11]);

	//selectam doar primii K vectori proprii
	MatrixXd U_K = U.block(0, 0, length, K);

	// reprezentam fetele initiale prin combinatii liniare ale vectorilor proprii
	MatrixXd coeff_faces(K, NR_FACES);
	for (int i = 0; i < NR_FACES; i++)
		coeff_faces.col(i) = U_K.colPivHouseholderQr().solve(eMatrix.col(i));

	// citim o alta imagine ce nu se afla in setul initial cu o persoana ce se afla in set si calculam coeficientii
	Mat input_face = imread("Images/person4/face2.jpg", IMREAD_GRAYSCALE);

	// aplicam acelasi algoritm
	uchar** input_array = new uchar *;
	input_array[0] = input_face.isContinuous() ? input_face.data : input_face.clone().data;

	double** input_matrix = getMatrix(input_array, mean_array, length, 1);
	MatrixXd input_eMatrix = ConvertToEigenMatrix(input_matrix, length, 1);
	
	MatrixXd coeff_input(K, 1);
	coeff_input.col(0) = U_K.colPivHouseholderQr().solve(input_eMatrix.col(0));

	// gasim distanta euclidiana minima intre vectorul ce contine coeficientii fetei citite
	// si vectorii ce contin coeficientii fetelor din setul initiale
	Index min_index;
	(coeff_faces.colwise() - coeff_input.col(0)).colwise().squaredNorm().minCoeff(&min_index);

	Mat output_face = faces_img[min_index];

	// afisam fata - input, iar output-ul este reprezentat de fata corespunzatoare distantei minime
	// adica fata recunoscuta din setul initial
	Mat input_color, output_color;
	cvtColor(input_face, input_color, COLOR_GRAY2RGB);
	cvtColor(output_face, output_color, COLOR_GRAY2RGB);
	ShowManyImages("Input Face + Recognized Face", 2, input_color, output_color);

	waitKey();
	return 0;
}
