/* Made by Alexander Rai */

#include <vector>
#include <random>
#include <chrono>
#include <iostream>

class Matrix
{
	std::vector<std::vector<double>> matrix;

	void swap_rows(std::vector<std::vector<double>>& matrix, int first_row, int second_row);
	void swap_columns(std::vector<std::vector<double>>& matrix, int first_column, int second_column);
public:
	Matrix(int matrix_size, int N);
	Matrix(const std::vector<std::vector<double>>& matrix_to_copy);
	Matrix(const Matrix& matrix_to_copy);

	static std::vector<double> dot_vector_simmetric(const std::vector<std::vector<double>>& dot_matrix, 
		const std::vector<double>& vector); /* multiplies simmetric matrix by vector */
	static std::vector<double> dot_vector_sub_matrix(const std::vector<std::vector<double>>& dot_matrix, int n, int m,
		const std::vector<double>& vector); /* multiplies sub matrix by vector */
	Matrix get_inverse_matrix(); /* returns inverse matrix */
	double get_matrix_lrate();	/* returns matrix l rate */
	double get_condition_number(); /* returns condition number of the matrix */
	std::vector<double> solve_gauss_matrix(std::vector<double> b); /* solves equations and retuns answers vector */
	std::vector<double> solve_l_lt_matrix(std::vector<double> b); /* solves equations and retuns answers vector */
	std::vector<double> solve_normal_method(std::vector<double> b, int N); /* solves equation and returns vector */
	std::vector<double> solve_relaxation_method(std::vector<double> b, double w);
	std::vector<double> solve_l_u_p(std::vector<double>& b);
	std::vector<double> get_next_gauss_zeidel(const std::vector<std::vector<double>> &B,
		const std::vector<double> &previous_x,
		const std::vector<double> &g);

	std::vector<std::vector<double>> get_l_lt(); /* retruns LL^t as one matrix */
	
	
	//std::vector<std::vector<double>> get_submatrix(int n, int m); /* returns submatrix size of n x m */


	static std::vector<double> get_random_vector(int vector_size, int N); /* returns randomly filled vector */
	static Matrix get_unit_matrix(int matrix_size); /* returns unit matrix size of matrix_size */

	std::vector<std::vector<double>>& get_as_vector(); /* returns Matrix as vector of vectors */

};


/**
* Сделать:
* - обратная матрица и метод Гаусса для симметричных
*/