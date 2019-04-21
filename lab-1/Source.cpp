#include <iostream>
#include "Matrix.h"

#define ASIZE 256
#define NRANDOM 4
void print_matrix(Matrix matrix, std::vector<double> b);
double get_euql_sq(const std::vector<double>& a,const std::vector<double>& b);
double get_difference_max_rate(const std::vector<double>& a, const std::vector<double>& b);

int main()
{
	for (int i = 0; i < 2; i++)
	{
		Matrix matrix(ASIZE, NRANDOM);
		auto matrix_A = matrix.get_as_vector();
		auto vector_y = Matrix::get_random_vector(ASIZE, NRANDOM);
	
		//auto vector_b = matrix.dot_vector(vector_y);
		//std::cout << matrix.get_matrix_lrate() << std::endl;
		//std::cout << "Condition number: " << matrix.get_condition_number() << std::endl;
		auto b = matrix.dot_vector_simmetric(matrix.get_as_vector(), vector_y);
		//print_matrix(matrix, b);
		//auto answers = matrix.solve_normal_method(b, NRANDOM * 20); // 20 * N for normal method
		auto answers = matrix.solve_l_u_p(b);
		//std::cout << "Euqlid: " << get_euql_sq(Matrix::dot_vector_sub_matrix(matrix_A, ASIZE, NRANDOM * 20, answers), b) << "\nAnswers:\n";
		std::cout << "Max difference: " << get_difference_max_rate(answers, vector_y) << std::endl;
		
		for (int i = 0; i < answers.size(); i++)
			std::cout << answers[i] << " ";
		std::cout << std::endl;
		for (int i = 0; i < vector_y.size(); i++)
			std::cout << vector_y[i] << " ";
		//print_matrix(matrix.get_inverse_matrix());
		std::cout << "\n------------------------------------\n";
	}
	return 0;
}

double get_euql_sq(const std::vector<double>& a, const std::vector<double>& b)
{
	double result = 0;
	for (int i = 0; i < a.size(); i++)
	{
		result += std::pow(a[i] - b[i], 2);
	}

	return sqrt(result);
}

double get_difference_max_rate(const std::vector<double>& a, const std::vector<double>& b)
{
	double max_difference = 0;

	for (int i = 0; i < a.size(); i++)
		if (abs(a[i] - b[i]) > max_difference)
			max_difference = abs(a[i] - b[i]);

	return max_difference;
}

void print_matrix(Matrix matrix, std::vector<double> b)
{
	auto vector_matrix = matrix.get_as_vector();
	for (int i = 0; i < vector_matrix.size(); i++)
	{
		for (int j = 0; j < vector_matrix.size(); j++)
		{
			std::cout << vector_matrix[i][j] << " ";
		}
		std::cout << " | " << b[i] << std::endl;
	}
}