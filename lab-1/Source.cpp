#include <iostream>
#include <fstream>
#include <windows.h>
#include "Matrix.h"

#define ASIZE 256
#define NRANDOM 4
#define NTEST 100

double get_euql_diff_sq(const std::vector<double>& a,const std::vector<double>& b); // returns euqlid square norm
																					// of vectors difference
double get_difference_max_rate(const std::vector<double>& a, const std::vector<double>& b); // returns max norm
																							// of vectors difference
double *get_min_max_average(const std::vector<double>& a); // returns an array of 3 elements(min, max and average) of vector

int main()
{
	std::vector<double> condition_numbers(NTEST, 0);
	std::vector<double> gauss_max_norms(NTEST, 0);
	std::vector<double> lup_max_norms(NTEST, 0);
	std::vector<double> l_lt_max_norms(NTEST, 0);
	std::vector<double> relaxation_max_norms(NTEST, 0);
	std::vector<double> reflection_max_norms(NTEST, 0);
	std::vector<double> norm_square_max_norm(NTEST, 0);
	std::vector<double> gmres_max_norms(NTEST, 0);
	std::vector<double> arnoldi_max_norms(NTEST, 0);

	for (int i = 0; i < NTEST; i++)
	{
		std::cout << "Iteration #" << i + 1;
		Matrix matrix(ASIZE, NRANDOM);
		auto matrix_A = matrix.get_as_vector();
		auto y_vector = Matrix::get_random_vector(ASIZE, NRANDOM);
		auto b_vector = Matrix::dot_vector_simmetric(matrix_A, y_vector);
		/* condition number testing */
		condition_numbers[i] = matrix.get_condition_number();
		std::cout << "\nCondition number: " << condition_numbers[i];
		/* gauss method testing */
		{
			auto answers = matrix.solve_gauss_matrix(b_vector);
			gauss_max_norms[i] = get_difference_max_rate(y_vector, answers);
			std::cout << "\nGauss: " << gauss_max_norms[i];
		}
		/* lup method testing */
		{
			auto answers = matrix.solve_l_u_p(b_vector);
			lup_max_norms[i] = get_difference_max_rate(y_vector, answers);
			std::cout << "\nLUP: " << lup_max_norms[i];
		}
		/* square root method testing */
		{
			auto answers = matrix.solve_square_root_matrix(b_vector);
			l_lt_max_norms[i] = get_difference_max_rate(y_vector, answers);
			std::cout << "\nSquare root: " << l_lt_max_norms[i];
		}
		/* reflection method testing */
		{
			auto answers = matrix.solve_reflection_method(b_vector);
			reflection_max_norms[i] = get_difference_max_rate(y_vector, answers);
			std::cout << "\nReflection: " << reflection_max_norms[i];
		}
		/* relaxation method testing */
		{
			auto answers = matrix.solve_relaxation_method(b_vector, double(NRANDOM + 1) / 6, 1e-15);
			relaxation_max_norms[i] = get_difference_max_rate(y_vector, answers);
			std::cout << "\nRelaxation: " << relaxation_max_norms[i];
		}
		/* least squares problem testing */
		{
			auto answers = matrix.solve_normal_method(b_vector, NRANDOM * 20);
			norm_square_max_norm[i] = get_euql_diff_sq(Matrix::dot_vector_sub_matrix(matrix_A, ASIZE, NRANDOM * 20, answers), b_vector);
			std::cout << "\nNorm: " << norm_square_max_norm[i];
		}
		/* GMRES method testing */
		{
			auto answers = matrix.solve_gmres(b_vector, 1e-3);
			gmres_max_norms[i] = get_difference_max_rate(y_vector, answers);
			std::cout << "\nGMRES: " << gmres_max_norms[i];
		}
		/* arnoldi method testing */
		{
			auto answers = matrix.solve_arnoldi(b_vector, 1e-3);
			arnoldi_max_norms[i] = get_difference_max_rate(y_vector, answers);
			std::cout << "\nGMRES(Arnoldi): " << arnoldi_max_norms[i];
			
		}
		std::cout << "\n\n";
	}

	std::ofstream fout("report.txt", std::ios::out);

	double *min_max_average = get_min_max_average(condition_numbers);
	fout << "Размер матрицы : " << ASIZE
		<< "\nЧисло N = " << NRANDOM
		<< "\nКол-во тестов: " << NTEST
		<< "\n----------------------------------------\n";

	fout << "Минимальное, максимальное и средннее число обусловленности: " 
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время нахождения обратной матрицы: " << (double)Matrix::time_inverse_matrix / NTEST << " мс\n\n";

	min_max_average = get_min_max_average(gauss_max_norms);
	fout << "Минимальная, максимальная и среднняя норма разности решения методом Гаусса: " 
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время решения методом Гаусса: " << (double)Matrix::general_gauss_time / NTEST << " мс\n\n";

	min_max_average = get_min_max_average(lup_max_norms);
	fout << "Минимальная, максимальная и среднняя норма разности решения методом LUP-разложения: " 
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время решения методом LUP-разложения: " << (double)Matrix::general_lup_solving_time / NTEST << " мс\n"
		<< "Среднее время построения LUP-разложения: " << (double)Matrix::general_lup_creating_time / NTEST << " мc\n\n";

	min_max_average = get_min_max_average(l_lt_max_norms);
	fout << "Минимальная, максимальная и среднняя норма разности решения методом квадратного корня: " 
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время решения методом квадратного корня: " << (double)Matrix::general_l_lt_time / NTEST << " мс\n\n";

	min_max_average = get_min_max_average(relaxation_max_norms);
	fout << "Минимальная, максимальная и среднняя норма разности решения методом релаксации: "
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время решения методом релаксации: " << (double)Matrix::general_relaxation_time / NTEST << " мс\n\n";

	min_max_average = get_min_max_average(reflection_max_norms);
	fout << "Минимальная, максимальная и среднняя норма разности решения методом отражений: "
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время решения методом отражений: " << (double)Matrix::general_reflection_time / NTEST << " мс\n\n";

	min_max_average = get_min_max_average(norm_square_max_norm);
	fout << "Минимальная, максимальная и среднняя норма разности решения задачи наименьших квадратов: "
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время решения методом наименьших квадратов: " << (double)Matrix::general_norm_time / NTEST << " мс\n\n";

	min_max_average = get_min_max_average(gmres_max_norms);
	fout << "Минимальная, максимальная и среднняя норма разности решения GMRES: "
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время решения методом GMRES: " << (double)Matrix::general_gmres_time / NTEST << " мс\n\n";
	
	min_max_average = get_min_max_average(arnoldi_max_norms);
	fout << "Минимальная, максимальная и среднняя норма разности решения GMRES(Арнольди): "
		<< min_max_average[0] << " " << min_max_average[1] << " " << min_max_average[2] << "\n"
		<< "Среднее время решения методом GMRES(Арнольди): " << (double)Matrix::general_arnoldi_time / NTEST << " мс\n\n";

	fout.close();
	return 0;
}

double get_euql_diff_sq(const std::vector<double>& a, const std::vector<double>& b)
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
		if (fabs(a[i] - b[i]) > max_difference)
			max_difference = fabs(a[i] - b[i]);

	return max_difference;
}

double *get_min_max_average(const std::vector<double>& a)
{
	double *min_max_average = new double[3]{};
	min_max_average[0] = LLONG_MAX;

	for (int i = 0; i < a.size(); i++)
	{
		if (a[i] < min_max_average[0])
			min_max_average[0] = a[i];

		if (a[i] > min_max_average[1])
			min_max_average[1] = a[i];

		min_max_average[2] += a[i];
	}
	min_max_average[2] /= a.size();
	return min_max_average;
}