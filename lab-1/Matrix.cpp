#include "Matrix.h"

Matrix::Matrix(int matrix_size, int N)
{
	std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<> urd(-pow(2, N / 4), pow(2, N / 4));

	matrix = std::vector<std::vector<double>>(matrix_size, std::vector<double>(matrix_size, 0));

	std::vector<double> diagonal_elements(matrix_size, 0);

	for (int i = 0; i < matrix_size; i++)
	{
		for (int j = i + 1; j < matrix_size; j++)
		{
			matrix[i][j] = matrix[j][i] = urd(rng);
			diagonal_elements[i] += std::abs(matrix[i][j]);
			diagonal_elements[j] += std::abs(matrix[i][j]);
		}
		matrix[i][i] = diagonal_elements[i];
	}

}

Matrix::Matrix(const std::vector<std::vector<double>>& matrix_to_copy) : matrix(matrix_to_copy) {}

Matrix::Matrix(const Matrix& matrix_to_copy) : matrix(matrix_to_copy.matrix) {}

std::vector<double> Matrix::dot_vector(const std::vector<double>& vector)
{
	// add chaecking for uncoordination
	std::vector<double> result_vector(vector.size(), 0);

	for (int i = 0; i < matrix.size(); i++)
	{
		result_vector[i] += matrix[i][i] * vector[i];
		for (int j = i + 1; j < matrix.size(); j++)
		{
			result_vector[i] += matrix[i][j] * vector[j];
			result_vector[j] += matrix[i][j] * vector[i];
		}
	}

	return result_vector;
}

Matrix Matrix::get_inverse_matrix() // what if diagonal element is 0?
{
	auto inverse_matrix = get_unit_matrix(matrix.size()).get_as_vector();
	auto real_matrix = matrix;
	/* forward pass */
	for (int i = 0; i < real_matrix.size() - 1; i++)
	{
		for (int j = i + 1; j < real_matrix.size(); j++)
		{
			double element_coefficient = real_matrix[j][i] / real_matrix[i][i];

			real_matrix[j][i] = 0;
			inverse_matrix[j][i] -= inverse_matrix[i][i] * element_coefficient;

			for (int k = i + 1; k < real_matrix.size(); k++)
			{
				real_matrix[j][k] -= real_matrix[i][k] * element_coefficient;
				inverse_matrix[j][k] -= inverse_matrix[i][k] * element_coefficient;
			}
		}
	}
	/* reverse pass */
	for (int i = real_matrix.size() - 1; i >= 0; i--)
	{
		for (int k = 0; k < real_matrix.size(); k++)
			inverse_matrix[i][k] /= real_matrix[i][i];

		real_matrix[i][i] = 1;

		for (int j = i - 1; j >= 0; j--)
		{
			for (int k = 0; k < real_matrix.size(); k++)
				inverse_matrix[j][k] -= inverse_matrix[i][k] * real_matrix[j][i];

			real_matrix[j][i] = 0;
		}
	}

	return Matrix(inverse_matrix);
}

double Matrix::get_matrix_lrate()
{
	double row_max = 0;

	for (int i = 0; i < matrix.size(); i++)
	{
		double current_row_max = 0;

		for (int j = 0; j < matrix.size(); j++)
			current_row_max += std::abs(matrix[i][j]);

		if (current_row_max > row_max)
			row_max = current_row_max;
	}

	return row_max;
}

double Matrix::get_condition_number()
{
	return get_matrix_lrate() * get_inverse_matrix().get_matrix_lrate();
}

std::vector<double> Matrix::get_random_vector(int vector_size, int N)
{
	std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<> urd(pow(2, -N / 4), pow(2, N / 4));

	std::vector<double> vector(vector_size);

	for (int i = 0; i < vector_size; i++)
		vector[i] = urd(rng);

	return vector;
}

std::vector<double> Matrix::get_answers(std::vector<double> b) // add chaecking for uncoordination
{
	std::vector<double> answers(matrix.size(), 0);
	auto real_matrix = matrix;

	int *vertical_permutation = new int[matrix.size()];
	int *horizontal_permutation = new int[matrix.size()];

	for (int i = 0; i < matrix.size(); i++)
	{
		vertical_permutation[i] = i;
		horizontal_permutation[i] = i;
	}
	/* forward pass */
	for (int i = 0; i < real_matrix.size() - 1; i++)
	{
		double max_element = 0;
		int i_max_element = 0, j_max_element = 0;
		for (int k = i; k < real_matrix.size(); k++) /* choosing of heading element */
		{
			for (int p = i; p < real_matrix.size(); p++)
			{
				if (std::abs(real_matrix[vertical_permutation[k]][horizontal_permutation[p]]) > max_element)
				{
					max_element = std::abs(real_matrix[vertical_permutation[k]][horizontal_permutation[p]]);
					i_max_element = k;
					j_max_element = p;
				}
			}
		}

		std::swap(vertical_permutation[i], vertical_permutation[i_max_element]);
		std::swap(horizontal_permutation[i], vertical_permutation[j_max_element]);

		for (int j = i + 1; j < real_matrix.size(); j++)
		{
			double element_coefficient = real_matrix[vertical_permutation[j]][horizontal_permutation[i]]
				/ real_matrix[vertical_permutation[i]][horizontal_permutation[i]];

			real_matrix[vertical_permutation[j]][horizontal_permutation[i]] = 0;
			b[vertical_permutation[j]] -= b[vertical_permutation[i]] * element_coefficient;

			for (int k = i + 1; k < real_matrix.size(); k++)
				real_matrix[vertical_permutation[j]][horizontal_permutation[k]] -=
				real_matrix[vertical_permutation[i]][horizontal_permutation[k]] * element_coefficient;

		}
		std::cout << "\n======\n";
		for (int i = 0; i < real_matrix.size(); i++)
		{
			for (int j = 0; j < real_matrix.size(); j++)
			{
				std::cout << real_matrix[vertical_permutation[i]][horizontal_permutation[j]] << " ";
			}
			std::cout << " | " << b[vertical_permutation[i]] << std::endl;
		}
	}
	std::cout << "\n======\n";
	for (int i = 0; i < real_matrix.size(); i++)
	{
		for (int j = 0; j < real_matrix.size(); j++)
		{
			std::cout << real_matrix[vertical_permutation[i]][horizontal_permutation[j]] << " ";
		}
		std::cout << " | " << b[vertical_permutation[i]] << std::endl;
	}
	/* reverse pass */
	for (int i = real_matrix.size() - 1; i >= 0; i--)
	{

		b[vertical_permutation[i]] /= real_matrix[vertical_permutation[i]][horizontal_permutation[i]];

		real_matrix[vertical_permutation[i]][horizontal_permutation[i]] = 1;

		for (int j = i - 1; j >= 0; j--)
		{
			b[vertical_permutation[j]] -= b[vertical_permutation[i]] * real_matrix[vertical_permutation[j]][horizontal_permutation[i]];
			real_matrix[vertical_permutation[j]][horizontal_permutation[i]] = 0;
		}
	}
	std::cout << "\n======\n";
	for (int i = 0; i < real_matrix.size(); i++)
	{
		for (int j = 0; j < real_matrix.size(); j++)
		{
			std::cout << real_matrix[vertical_permutation[i]][horizontal_permutation[j]] << " ";
		}
		std::cout << " | " << b[vertical_permutation[i]] << std::endl;
	}
	for (int i = 0; i < answers.size(); i++)
		std::swap(answers[i], answers[horizontal_permutation[i]]);


	return answers;
}

Matrix Matrix::get_unit_matrix(int matrix_size)
{
	std::vector<std::vector<double>> unit_matrix(matrix_size, std::vector<double>(matrix_size, 0));

	for (int i = 0; i < unit_matrix.size(); i++)
		unit_matrix[i][i] = 1;

	return Matrix(unit_matrix);
}

std::vector<std::vector<double>>& Matrix::get_as_vector()
{
	return matrix;
}