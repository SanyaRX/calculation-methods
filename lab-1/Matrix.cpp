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

std::vector<double> Matrix::dot_vector_simmetric(const std::vector<std::vector<double>>& dot_matrix,
	const std::vector<double>& vector)
{
	std::vector<double> result_vector(dot_matrix.size(), 0);

	for (int i = 0; i < dot_matrix.size(); i++)
	{
		result_vector[i] += dot_matrix[i][i] * vector[i];
		for (int j = i + 1; j < dot_matrix[i].size(); j++)
		{
			result_vector[i] += dot_matrix[i][j] * vector[j];
			result_vector[j] += dot_matrix[i][j] * vector[i];
		}
	}

	return result_vector;
}

std::vector<double> Matrix::dot_vector_sub_matrix(const std::vector<std::vector<double>>& dot_matrix, int n, int m,
	const std::vector<double>& vector)
{
	std::vector<double> result_vector(n, 0);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			result_vector[i] += dot_matrix[i][j] * vector[j];
		}
	}

	return result_vector;
}

Matrix Matrix::get_inverse_matrix()
{
	auto inverse_matrix = get_unit_matrix(matrix.size()).get_as_vector();
	auto real_matrix = matrix;
	/* forward pass */
	for (int i = 0; i < real_matrix.size() - 1; i++)
	{
		if (real_matrix[i][i] == 0)
		{
			int j = i + 1;
			for (; j < real_matrix.size(); j++)
				if (real_matrix[j][i] != 0)
					break;

			swap_rows(real_matrix, i, j);
			swap_rows(inverse_matrix, i, j);
		}

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
	std::uniform_real_distribution<> urd(-pow(2, N / 4), pow(2, N / 4));

	std::vector<double> vector(vector_size);

	for (int i = 0; i < vector_size; i++)
		vector[i] = urd(rng);

	return vector;
}

void Matrix::swap_rows(std::vector<std::vector<double>>& matrix, int first_row, int second_row) 
{
	for (int i = 0; i < matrix.size(); i++) // rows swap
	{
		double temp = matrix[second_row][i];
		matrix[second_row][i] = matrix[first_row][i];
		matrix[first_row][i] = temp;
	}
}

void Matrix::swap_columns(std::vector<std::vector<double>>& matrix, int first_column, int second_column)
{
	for (int i = 0; i < matrix.size(); i++) // colums swap
	{
		double temp = matrix[i][second_column];
		matrix[i][second_column] = matrix[i][first_column];
		matrix[i][first_column] = temp;
	}
}

std::vector<double> Matrix::solve_gauss_matrix(std::vector<double> b) // add chaecking for uncoordination
{
	std::vector<double> answers(matrix.size(), 0);
	auto real_matrix = matrix;

	int *horizontal_permutation = new int[matrix.size()];

	for (int i = 0; i < matrix.size(); i++)
	{
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
				if (std::abs(real_matrix[k][p]) > max_element)
				{
					max_element = std::abs(real_matrix[k][p]);
					i_max_element = k;
					j_max_element = p;
				}
			}
		}

		std::swap(horizontal_permutation[i], horizontal_permutation[j_max_element]);

		std::swap(b[i], b[i_max_element]);

		swap_rows(real_matrix, i, j_max_element);
		swap_columns(real_matrix, i, i_max_element);
		for (int j = i + 1; j < real_matrix.size(); j++)
		{
			double element_coefficient = real_matrix[j][i] / real_matrix[i][i];

			real_matrix[j][i] = 0;
			b[j] -= b[i] * element_coefficient;

			for (int k = i + 1; k < real_matrix.size(); k++)
				real_matrix[j][k] -= real_matrix[i][k] * element_coefficient;
		}
	}
	/* reverse pass */
	
	for (int i = real_matrix.size() - 1; i >= 0; i--)
	{
		double sum = 0;
		for (int j = i + 1; j < real_matrix.size(); j++)
			sum += real_matrix[i][j] * b[j];

		b[i] = (b[i] - sum) / real_matrix[i][i];
	}

	for (int i = 0; i < matrix.size(); i++)
		answers[horizontal_permutation[i]] = b[i];
		
	return answers;
}

std::vector<std::vector<double>> Matrix::get_l_lt()
{
	auto matrix_l_lt = matrix;

	for (int i = 0; i < matrix_l_lt.size(); i++)  // forward pass
	{
		double sqrt = std::sqrt(matrix_l_lt[i][i]);
		matrix_l_lt[i][i] = sqrt;
		for(int j = i + 1; j < matrix_l_lt.size(); j++)
			matrix_l_lt[i][j] /= sqrt;
		
		for (int j = i + 1; j < matrix_l_lt.size(); j++)
		{
			double element_coefficient = matrix_l_lt[j][i] / matrix_l_lt[i][i];
			matrix_l_lt[j][i] = 0;

			for (int k = i + 1; k < matrix_l_lt.size(); k++)
			{
				matrix[j][k] -= matrix[i][k] * element_coefficient;
			}
		}
	}

	return matrix_l_lt;
}

std::vector<double> Matrix::solve_l_lt_matrix(std::vector<double> b) // works but not so good
{
	auto matrix_lt = get_l_lt();

	for (int i = 0; i < matrix_lt.size(); i++)
	{
		double sum = 0;
		
		for (int j = i - 1; j >= 0; j--)
			sum += matrix_lt[i][j] * b[j];

		b[i] = (b[i] - sum) / matrix_lt[i][i];
	}

	for (int i = matrix_lt.size() - 1; i >= 0; i--)
	{
		double sum = 0;

		for (int j = i + 1; j < matrix_lt.size(); j++)
			sum += matrix_lt[i][j] * b[j];

		b[i] = (b[i] - sum) / matrix_lt[i][i];
	}
	

	return b;
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

std::vector<double> Matrix::solve_normal_method(std::vector<double> b, int N)// what is the answer?
{
	std::vector<std::vector<double>> a_matrix(N, std::vector<double>(N, 0));
	std::vector<double> b_vector(N, 0);

	for (int i = 0; i < N; i++)
	{
		for(int k = 0; k < matrix.size(); k++)
			b_vector[i] += matrix[k][i] * b[k];
		for (int j = 0; j < N; j++)
		{
			
			for (int k = 0; k < matrix.size(); k++)
			{
				a_matrix[i][j] += matrix[k][i] * matrix[k][j];
			}
		}
	}

	for (int i = 0; i < a_matrix.size() - 1; i++)
	{
		for (int j = i + 1; j < a_matrix.size(); j++)
		{
			double element_coefficient = a_matrix[j][i] / a_matrix[i][i];

			a_matrix[j][i] = 0;
			b_vector[j] -= b_vector[i] * element_coefficient;

			for (int k = i + 1; k < a_matrix.size(); k++)
				a_matrix[j][k] -= a_matrix[i][k] * element_coefficient;
		}
	}
	/* reverse pass */

	for (int i = a_matrix.size() - 1; i >= 0; i--)
	{
		double sum = 0;

		for (int j = i + 1; j < a_matrix.size(); j++)
			sum += a_matrix[i][j] * b_vector[j];

		b_vector[i] = (b_vector[i] - sum) / a_matrix[i][i];
	}

	return b_vector;
}

std::vector<double> Matrix::get_next_gauss_zeidel(const std::vector<std::vector<double>> &B, const std::vector<double> &previous_x,
	const std::vector<double> &g)
{
	auto next_x = previous_x;
	for (int i = 0; i < B.size(); i++)
	{
		double temp = 0;
		for(int j = 0; j < B.size(); j++)
		{
			temp += B[i][j] * next_x[j];
		}
		next_x[i] = temp + g[i];
	}
	return next_x;
}

std::vector<double> Matrix::solve_relaxation_method(std::vector<double> b, double w)
{
	auto B = matrix;
	for (int i = 0; i < B.size(); i++)
	{
		double temp = B[i][i];
		b[i] /= temp;
		B[i][i] = 0;
		for (int j = 0; j < B.size(); j++)
			B[i][j] /= -temp;
	}
	auto x = b;
	for (int i = 0; i < 100; i++) // change size
	{
		auto next_x = get_next_gauss_zeidel(B, x, b);
		for (int j = 0; j < next_x.size(); j++)
		{
			x[j] = (1 - w) * x[j] + w * next_x[j];
		}
	}

	return x;
}

std::vector<double> Matrix::solve_l_u_p(std::vector<double>& b) // doesn't work
{
	auto matrix_l_u = matrix;
	auto b_shtrih = b;
	
	for (int i = 0; i < matrix_l_u.size() - 1; i++)
	{
		double max_element = 0;
		int j_max;
		for (int j = 0; j < matrix_l_u.size(); j++)
		{
			if (std::abs(matrix[j][i]) > max_element)
			{
				max_element = std::abs(matrix[j][i]);
				j_max = j;
			}
		}

		if (j_max != 0)
		{
			std::swap(b_shtrih[i], b_shtrih[j_max]);
			swap_rows(matrix_l_u, i, j_max);
		}

		for (int j = i + 1; j < matrix_l_u.size(); j++)
		{
			matrix_l_u[j][i] /= matrix_l_u[i][i];

			for (int k = i + 1; k < matrix_l_u.size(); k++)
				matrix_l_u[j][k] -= matrix_l_u[j][i] * matrix_l_u[i][k];
		}
	}

	for (int i = 1; i < matrix_l_u.size() - 1; i++)
	{
		double sum = 0;

		for (int j = i - 1; j >= 0; j--)
			sum += matrix_l_u[i][j] * b_shtrih[j];

		b_shtrih[i] -= sum;
	}

	
	for (int i = matrix_l_u.size() - 1; i >= 0; i--)
	{
		double sum = 0;

		for (int j = i + 1; j < matrix_l_u.size(); j++)
			sum += matrix_l_u[i][j] * b_shtrih[j];

		b_shtrih[i] = (b_shtrih[i] - sum) / matrix_l_u[i][i];
	}

	return b_shtrih;
}