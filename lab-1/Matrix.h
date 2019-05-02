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
	/* timers for testing */
	static long long general_gauss_time;
	static long long general_lup_creating_time;
	static long long general_lup_solving_time;
	static long long general_l_lt_time;
	static long long general_relaxation_time;
	static long long general_reflection_time;
	static long long general_norm_time;
	static long long general_gmres_time;
	static long long general_arnoldi_time;
	static long long time_inverse_matrix;

	Matrix(int matrix_size, int N);
	Matrix(const std::vector<std::vector<double>>& matrix_to_copy);
	Matrix(const Matrix& matrix_to_copy);

	static std::vector<double> dot_vector_simmetric(const std::vector<std::vector<double>>& dot_matrix, 
		const std::vector<double>& vector); /* multiplies simmetric matrix by vector */
	static std::vector<double> dot_vector_sub_matrix(const std::vector<std::vector<double>>& dot_matrix, int n, int m,
		const std::vector<double>& vector); /* multiplies sub matrix by vector */
	static double get_scalar_rpoduct(const std::vector<double>& a, const std::vector<double>& b);
	static double get_euqlid_norm(const std::vector<double> &b);
	static std::vector<double> get_difference_vectors(const std::vector<double>& a, const std::vector<double>& b);
	Matrix get_inverse_matrix(); /* returns inverse matrix */
	double get_matrix_lrate();	/* returns matrix l rate */
	double get_condition_number(); /* returns condition number of the matrix */
	
	std::vector<double> solve_gauss_matrix(std::vector<double> b); /* solves equations and retuns answers vector */
	std::vector<double> solve_square_root_matrix(std::vector<double> b); /* solves equations and retuns answers vector */
	std::vector<double> solve_normal_method(std::vector<double> b, int N); /* solves equation and returns vector */
	std::vector<double> solve_relaxation_method(std::vector<double> b, double w, double epsilon);
	std::vector<double> solve_l_u_p(std::vector<double> b);
	std::vector<double> solve_reflection_method(std::vector<double> b);
	std::vector<double> solve_gmres(std::vector<double> b, double epsilon);
	std::vector<double> solve_arnoldi(std::vector<double>& b, double epsilon);

	std::vector<std::vector<double>> get_l_lt(); /* retruns LEL^t as one matrix */


	static std::vector<double> get_random_vector(int vector_size, int N); /* returns randomly filled vector */
	static Matrix get_unit_matrix(int matrix_size); /* returns unit matrix size of matrix_size */

	std::vector<std::vector<double>>& get_as_vector(); /* returns Matrix as vector of vectors */

};
