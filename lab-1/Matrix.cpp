#include "Matrix.h"

long long Matrix::general_gauss_time = 0;
long long Matrix::general_lup_creating_time = 0;
long long Matrix::general_lup_solving_time = 0;
long long Matrix::general_l_lt_time = 0;
long long Matrix::general_relaxation_time = 0;
long long Matrix::general_reflection_time = 0;
long long Matrix::general_norm_time = 0;
long long Matrix::general_gmres_time = 0;
long long Matrix::general_arnoldi_time = 0;
long long Matrix::time_inverse_matrix = 0;


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
			diagonal_elements[i] += std::fabs(matrix[i][j]);
			diagonal_elements[j] += std::fabs(matrix[i][j]);
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
	auto start = std::chrono::system_clock::now();


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
	auto end = std::chrono::system_clock::now();
	time_inverse_matrix += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();

	return Matrix(inverse_matrix);
}

std::vector<double> Matrix::get_difference_vectors(const std::vector<double>& a, const std::vector<double>& b)
{
	std::vector<double> c(a.size(), 0);

	for (int i = 0; i < a.size(); i++)
		c[i] = a[i] - b[i];

	return c;
}

double Matrix::get_euqlid_norm(const std::vector<double> &b)
{
	double sum = 0;

	for (int i = 0; i < b.size(); i++)
		sum += std::pow(b[i], 2);

	return std::sqrt(sum);
}

double Matrix::get_scalar_rpoduct(const std::vector<double>& a, const std::vector<double>& b)
{
	double sum = 0;

	for (int i = 0; i < a.size(); i++)
		sum += a[i] * b[i];

	return sum;
}

double Matrix::get_matrix_lrate()
{
	double row_max = 0;

	for (int i = 0; i < matrix.size(); i++)
	{
		double current_row_max = 0;

		for (int j = 0; j < matrix.size(); j++)
			current_row_max += std::fabs(matrix[i][j]);

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
	
	auto start = std::chrono::system_clock::now();
	
	for (int i = 0; i < matrix.size(); i++)
	{
		horizontal_permutation[i] = i;
	}

	/* forward pass */
	for (int i = 0; i < real_matrix.size() - 1; i++)
	{
		double max_element = 0;
		int i_max_element = 0, j_max_element = 0;
		for (int k = i; k < real_matrix.size(); k++) // choosing of heading element
		{
			for (int p = i; p < real_matrix.size(); p++)
			{
				if (std::fabs(real_matrix[k][p]) > max_element)
				{
					max_element = std::fabs(real_matrix[k][p]);
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
	
	auto end = std::chrono::system_clock::now();
	general_gauss_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	
	return answers;
}

std::vector<std::vector<double>> Matrix::get_l_lt()
{
	auto matrix_l_lt = matrix;

	for (int i = 0; i < matrix_l_lt.size(); i++)  // forward pass
	{
		double sqrt = std::sqrt(fabs(matrix_l_lt[i][i]));
		matrix_l_lt[i][i] = matrix_l_lt[i][i] > 0 ? sqrt : -sqrt;
		for(int j = i + 1; j < matrix_l_lt.size(); j++)
			matrix_l_lt[i][j] /= sqrt;
		
		for (int j = i + 1; j < matrix_l_lt.size(); j++)
		{
			double element_coefficient = matrix_l_lt[j][i] / sqrt;
			matrix_l_lt[j][i] = 0;

			for (int k = i + 1; k < matrix_l_lt.size(); k++)
			{
				matrix_l_lt[j][k] -= matrix_l_lt[i][k] * element_coefficient;
			}
		}
	}

	return matrix_l_lt;
}

std::vector<double> Matrix::solve_square_root_matrix(std::vector<double> b) // works but not so good
{
	auto start = std::chrono::system_clock::now();
	auto matrix_lt = get_l_lt();

	for (int i = 0; i < matrix_lt.size(); i++)
	{
		double sum = 0;
		
		for (int j = i - 1; j >= 0; j--)
			sum += matrix_lt[j][i] * b[j];

		b[i] = (b[i] - sum) / fabs(matrix_lt[i][i]);
	}

	for (int i = 0; i < b.size(); i++)
		if (matrix_lt[i][i] < 0)
			b[i] *= -1;

	for (int i = matrix_lt.size() - 1; i >= 0; i--)
	{
		double sum = 0;

		for (int j = i + 1; j < matrix_lt.size(); j++)
			sum += matrix_lt[i][j] * b[j];

		b[i] = (b[i] - sum) / fabs(matrix_lt[i][i]);
	}
	
	auto end = std::chrono::system_clock::now();
	general_l_lt_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();

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

std::vector<double> Matrix::solve_normal_method(std::vector<double> b, int N)
{
	std::vector<std::vector<double>> a_matrix(N, std::vector<double>(N, 0));
	std::vector<double> b_vector(N, 0);
	auto start = std::chrono::system_clock::now();
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
			double element_coefficient = a_matrix[i][j] / a_matrix[i][i];
			b_vector[j] -= b_vector[i] * element_coefficient;

			for (int k = j; k < a_matrix.size(); k++)
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
	
	auto end = std::chrono::system_clock::now();
	general_norm_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	return b_vector;
}


std::vector<double> Matrix::solve_relaxation_method(std::vector<double> b, double w, double epsilon)
{
	auto B = matrix;

	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < B.size(); i++)
	{
		double temp = B[i][i];
		b[i] /= temp;
		B[i][i] = 0;
		for (int j = 0; j < B.size(); j++)
			B[i][j] /= -temp;
	}
	auto x = b;
	std::vector<double> prev_x;
	do
	{
		prev_x = x;
		for (int i = 0; i < x.size(); i++)
		{
			double sum = 0;
			
			for (int j = 0; j < x.size(); j++)
				sum += B[i][j] * x[j];

			x[i] = (1 - w) * x[i] + w * (b[i] + sum);
		}
	} while (get_euqlid_norm(get_difference_vectors(x, prev_x)) >= epsilon);

	auto end = std::chrono::system_clock::now();
	general_relaxation_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	return x;
}

std::vector<double> Matrix::solve_l_u_p(std::vector<double> b_shtrih) // works but not so good
{
	auto matrix_l_u = matrix;

	auto start_creating = std::chrono::system_clock::now();
	
	for (int i = 0; i < matrix_l_u.size() - 1; i++)
	{
		double max_element = 0;
		int j_max;
		for (int j = 0; j < matrix_l_u.size(); j++)
		{
			if (std::fabs(matrix_l_u[j][i]) > max_element)
			{
				max_element = std::fabs(matrix_l_u[j][i]);
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

	auto end_creating = std::chrono::system_clock::now();
	general_lup_creating_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end_creating - start_creating)).count();

	auto start = std::chrono::system_clock::now();
	for (int i = 1; i <= matrix_l_u.size() - 1; i++)
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
	auto end = std::chrono::system_clock::now();
	general_lup_solving_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	
	return b_shtrih;
}


std::vector<double> Matrix::solve_reflection_method(std::vector<double> b)
{
	auto a_matrix = matrix;
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < a_matrix.size() - 1; i++)
	{
		std::vector<double> a_shtrih(a_matrix.size() - i, 0);
		std::vector<double> w_vector(a_matrix.size() - i, 0);

		for (int j = i; j < a_matrix.size(); j++)
			a_shtrih[j - i] = a_matrix[j][i];
		
		a_matrix[i][i] = get_euqlid_norm(a_shtrih);


		a_shtrih[0] -= a_matrix[i][i];
		double norm = get_euqlid_norm(a_shtrih);

		for (int j = 0; j < w_vector.size(); j++)
			w_vector[j] = a_shtrih[j] / norm;

		for (int j = i + 1; j < a_matrix.size(); j++)
		{
			double scalar_product = 0;

			for (int k = i; k < a_matrix.size(); k++)
				scalar_product += a_matrix[k][j] * w_vector[k - i];

			for (int k = i; k < a_matrix.size(); k++)
				a_matrix[k][j] -= 2 * scalar_product * w_vector[k - i];

		}

		double scalar_product = 0;

		for (int k = i; k < a_matrix.size(); k++)
			scalar_product += b[k] * w_vector[k - i];
			
		for (int k = i; k < a_matrix.size(); k++)
			b[k] -= 2 * scalar_product * w_vector[k - i];

	}

	for (int i = a_matrix.size() - 1; i >= 0; i--)
	{	
		double sum = 0;

		for (int j = i + 1; j < a_matrix.size(); j++)
			sum += a_matrix[i][j] * b[j];

		b[i] = (b[i] - sum) / a_matrix[i][i];
	}
	auto end = std::chrono::system_clock::now();
	general_reflection_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	return b;
}


std::vector<double> Matrix::solve_gmres(std::vector<double> b, double epsilon)
{
	auto a_matrix = matrix;
	std::vector<std::vector<double>> kr_space;
	std::vector<double> answers;
	
	auto start = std::chrono::system_clock::now();
	kr_space.push_back(b);
	do
	{
		std::vector<std::vector<double>> a_dot_k(kr_space.size(), std::vector<double>(kr_space.size(), 0));
		std::vector<double> a_dot_b(kr_space.size(), 0);
		kr_space.push_back(dot_vector_simmetric(a_matrix, kr_space[kr_space.size() - 1]));

		for (int i = 1; i < kr_space.size(); i++)
		{
			for(int j = 0; j < b.size(); j++)
				a_dot_b[i - 1] += kr_space[i][j] * b[j];

			for (int j = 0; j < kr_space.size() - 1; j++)
			{
				for (int k = 0; k < b.size(); k++)
				{
					a_dot_k[i - 1][j] += kr_space[i][k] * kr_space[j + 1][k];
				}
			}
		}

		auto y_vector = Matrix(a_dot_k).solve_gauss_matrix(a_dot_b);

		answers = std::vector<double>(b.size(), 0);

		for (int i = 0; i < b.size(); i++)
		{
			for (int j = 0; j < kr_space.size() - 1; j++)
				answers[i] += kr_space[j][i] * y_vector[j];
		}
		

	} while (get_euqlid_norm(get_difference_vectors(dot_vector_simmetric(a_matrix, answers), b)) > epsilon);
	auto end = std::chrono::system_clock::now();
	general_gmres_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	return answers;
}

std::vector<double> Matrix::solve_arnoldi(std::vector<double>& b, double epsilon)
{
	std::vector<std::vector<double>> q_matrix;
	std::vector<std::vector<double>> h_matrix(matrix.size(), std::vector<double>(matrix.size(), 0));
	std::vector<double> answers;
	
	auto start = std::chrono::system_clock::now();

	double norm = get_euqlid_norm(b);
	q_matrix.push_back(b);
	
	for (int i = 0; i < q_matrix[0].size(); i++)
		q_matrix[0][i] /= norm;

	for (int j = 0; j < matrix.size(); j++)
	{
		auto z = dot_vector_simmetric(matrix, q_matrix[j]);
		
		for (int i = 0; i < j + 1; i++)
		{
			h_matrix[i][j] = get_scalar_rpoduct(z, q_matrix[i]);
			
			for (int k = 0; k < z.size(); k++)
				z[k] -= h_matrix[i][j] * q_matrix[i][k];
		}

		h_matrix[j + 1][j] = get_euqlid_norm(z);
		
		if (h_matrix[j + 1][j] == 0)
			break;
		
		q_matrix.push_back(z);
		for (int k = 0; k < z.size(); k++)
			q_matrix[j + 1][k] /= h_matrix[j + 1][j];

		std::vector<std::vector<double>> h_dot_h_matrix(j + 1, std::vector<double>(j + 1, 0));
		std::vector<double> h_dot_d_vector(j + 1, 0);

		for (int i = 0; i < j + 1; i++)
		{
			h_dot_d_vector[i] = h_matrix[0][i] * norm;

			for (int k = 0; k < j + 1; k++)
			{
				for (int p = 0; p < j + 2; p++)
					h_dot_h_matrix[i][k] += h_matrix[p][i] * h_matrix[p][k];
			}
		}
		auto y_vector = Matrix(h_dot_h_matrix).solve_l_u_p(h_dot_d_vector);
		answers = std::vector<double>(q_matrix[0].size() , 0);

		for (int i = 0; i < answers.size(); i++)
		{
			for (int p = 0; p < y_vector.size(); p++)
				answers[i] += q_matrix[p][i] * y_vector[p];
		}
		
		if (get_euqlid_norm(get_difference_vectors(dot_vector_simmetric(matrix, answers), b)) < epsilon)
			break;
		
	}

	auto end = std::chrono::system_clock::now();
	general_arnoldi_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	return answers;
}