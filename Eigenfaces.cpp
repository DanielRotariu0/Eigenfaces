#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#define NR_FACES 9

using namespace cv;
using namespace std;
using namespace Eigen;

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

MatrixXd ConvertToEigenMatrix(double** data, int rows, int cols)
{
    MatrixXd eMatrix(rows, cols);
    for (int i = 0; i < rows; ++i)
        eMatrix.row(i) = VectorXd::Map(&data[i][0], cols);
    return eMatrix;
}

int main()
{
    // read faces

    Mat faces_img[NR_FACES];
    faces_img[0] = imread("Images/face0.png", IMREAD_GRAYSCALE);
    faces_img[1] = imread("Images/face1.png", IMREAD_GRAYSCALE);
    faces_img[2] = imread("Images/face2.png", IMREAD_GRAYSCALE);
    faces_img[3] = imread("Images/face3.png", IMREAD_GRAYSCALE);
    faces_img[4] = imread("Images/face4.png", IMREAD_GRAYSCALE);
    faces_img[5] = imread("Images/face5.png", IMREAD_GRAYSCALE);
    faces_img[6] = imread("Images/face6.png", IMREAD_GRAYSCALE);
    faces_img[7] = imread("Images/face7.png", IMREAD_GRAYSCALE);
    faces_img[8] = imread("Images/face8.png", IMREAD_GRAYSCALE);

    for (int i = 0; i < NR_FACES; i++)
        if (faces_img[i].empty()) {
            cout << "Image File Not Found" << endl;
            cin.get();
            return -1;
        }


    //get arrays from faces

    uchar** faces_array = new uchar * [NR_FACES];
    uint length = faces_img[0].total() * faces_img[0].channels();

    for (int i = 0; i < NR_FACES; i++)
        faces_array[i] = faces_img[i].isContinuous() ? faces_img[i].data : faces_img[i].clone().data;


    //for (int i = 0; i < length; i++)
        //cout << (double)faces_array[0][i] << " ";
    //cout << endl;
    //show faces
    /*
    imshow("Face0", faces_img[0]);
    imshow("Face1", faces_img[1]);
    imshow("Face2", faces_img[2]);
    imshow("Face3", faces_img[3]);
    imshow("Face4", faces_img[4]);
    imshow("Face5", faces_img[5]);
    imshow("Face6", faces_img[6]);
    imshow("Face7", faces_img[7]);
    imshow("Face8", faces_img[8]);
    */

    //imshow("Face0", faces_img[0]);

    //calculate mean array(mean face)
    uchar* mean_array = getMeanArray(faces_array, length);

    //array to mat
    Mat mean_face(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), mean_array);
    imshow("mean face", mean_face);

    //calculate matrix
    double** matrix = getMatrix(faces_array, mean_array, length);
    /*
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < NR_FACES; j++)
            cout << (double)matrix[i][j] << " ";
        cout << endl;
    }
    */


    //convert matrix to Eigen matrix
    MatrixXd eMatrix = ConvertToEigenMatrix(matrix, length, NR_FACES);
    //cout << eMatrix;
    // 
    //calculate covariance matrix
    MatrixXd covMatrix = eMatrix.transpose() * eMatrix;
    //cout << covMatrix << endl;

    //calculate eigenvectors
    EigenSolver<MatrixXd> s(covMatrix);

    MatrixXd D = s.pseudoEigenvalueMatrix();
    MatrixXd V = s.pseudoEigenvectors();
    cout << "The pseudo-eigenvalue matrix D is:" << endl << D << endl;
    cout << "The pseudo-eigenvector matrix V is:" << endl << V << endl;
    //cout << "Finally, V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;

    //map the eigenvectors into the C' using ui=A*vi
    MatrixXd V2 = eMatrix * V;
    //cout << "The matrix containing the highest M eigenvectors of C' is:" << endl << V2 << endl;
    cout << V2 << endl;


    //normalize the eigenvectors
   // for (int i = 0; i < NR_FACES; i++)
        //V2.col(i).normalize();

    //cout << V2;
    //extract the eigenvectors
    uchar** eigenvectors = new uchar * [NR_FACES];
    for (int i = 0; i < NR_FACES; i++)
        eigenvectors[i] = new uchar[length];

    for (int i = 0; i < NR_FACES; i++)
        for (int j = 0; j < length; j++)
            eigenvectors[i][j] = V2.col(i)[j];

    //??
    for (int i = 0; i < NR_FACES; i++)
        for (int j = 0; j < length; j++)
            if (faces_array[i][j] - mean_array[j] < 0)
                eigenvectors[i][j] = 0;
            else
                eigenvectors[i][j] = faces_array[i][j] - mean_array[j];

    Mat eigenface0(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[0]);
    imshow("eigenface0", eigenface0);
    Mat eigenface1(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[1]);
    imshow("eigenface1", eigenface1);
    Mat eigenface2(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[2]);
    imshow("eigenface2", eigenface2);
    Mat eigenface3(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[3]);
    imshow("eigenface4", eigenface3);
    Mat eigenface4(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[4]);
    imshow("eigenface4", eigenface4);
    Mat eigenface5(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[5]);
    imshow("eigenface5", eigenface5);
    Mat eigenface6(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[6]);
    imshow("eigenface6", eigenface6);
    Mat eigenface7(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[7]);
    imshow("eigenface7", eigenface7);
    Mat eigenface8(faces_img[0].rows, faces_img[0].cols, faces_img[0].type(), eigenvectors[8]);
    imshow("eigenface8", eigenface8);




    waitKey(0);
    return 0;
}