#include <iostream>
#include "Matrix.h"

#define ASIZE 256

void print_matrix(Matrix matrix, std::vector<double> b);

int main()
{
	Matrix matrix(3, 40);
	auto matrix_A = matrix.get_as_vector();
	auto vector_y = Matrix::get_random_vector(3, 40);
	//auto vector_b = matrix.dot_vector(vector_y);
	auto b = matrix.dot_vector(vector_y);
	print_matrix(matrix, b);
	auto answers = matrix.get_answers(b);
	for (int i = 0; i < answers.size(); i++)
		std::cout << answers[i] << " ";
	std::cout << std::endl;
	for (int i = 0; i < vector_y.size(); i++)
		std::cout << vector_y[i] << " ";
	//print_matrix(matrix.get_inverse_matrix());
	return 0;
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