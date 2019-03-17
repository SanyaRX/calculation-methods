/* Made by Alexander Rai */

#include <vector>
#include <random>
#include <chrono>
#include <iostream>

class Matrix
{
	std::vector<std::vector<double>> matrix;

public:
	Matrix(int matrix_size, int N);
	Matrix(const std::vector<std::vector<double>>& matrix_to_copy);
	Matrix(const Matrix& matrix_to_copy);

	std::vector<double> dot_vector(const std::vector<double>& vector); /* multiplies matrix by vector */
	Matrix get_inverse_matrix(); /* returns inverse matrix */
	double get_matrix_lrate();	/* returns matrix l rate */
	double get_condition_number(); /* returns condition number of the matrix */
	std::vector<double> get_answers(std::vector<double> b); /* solves equations and retuns answers vector */

	static std::vector<double> get_random_vector(int vector_size, int N); /* returns randomly filled vector */
	static Matrix get_unit_matrix(int matrix_size); /* returns unit matrix size of matrix_size */

	std::vector<std::vector<double>>& get_as_vector(); /* returns Matrix as vector of vectors */

};