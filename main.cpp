#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

using matrix = std::vector<std::vector<int>>;

constexpr int kNumThreads = 4;
constexpr int kMatrixSize = 1000;

// Утилиты для работы с матрицами
namespace matrix_utils
{
    inline size_t rows(const matrix &m) { return m.size(); }
    inline size_t columns(const matrix &m) { return !m.empty() ? m[0].size() : 0; }

    void print(const matrix &m)
    {
        for (const auto &row : m)
        {
            for (int val : row)
            {
                std::cout << val << " ";
            }
            std::cout << '\n';
        }
    }

    bool are_equal(const matrix &a, const matrix &b)
    {
        if (rows(a) != rows(b) || columns(a) != columns(b))
            return false;

        for (size_t i = 0; i < rows(a); ++i)
        {
            for (size_t j = 0; j < columns(a); ++j)
            {
                if (a[i][j] != b[i][j])
                    return false;
            }
        }
        return true;
    }

    matrix create(size_t rows, size_t cols, int value = 0)
    {
        return matrix(rows, std::vector<int>(cols, value));
    }
}

// Утилиты для замера времени
class Timer
{
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_;

public:
    Timer() : start_(clock::now()) {}

    void reset() { start_ = clock::now(); }

    double elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   clock::now() - start_)
            .count();
    }
};

// Последовательное умножение
matrix multiply_serial(const matrix &a, const matrix &b)
{
    assert(matrix_utils::columns(a) == matrix_utils::rows(b));

    const size_t a_rows = matrix_utils::rows(a);
    const size_t a_cols = matrix_utils::columns(a);
    const size_t b_cols = matrix_utils::columns(b);

    matrix result = matrix_utils::create(a_rows, b_cols);

    for (size_t i = 0; i < a_rows; ++i)
    {
        for (size_t k = 0; k < a_cols; ++k)
        {
            const int val = a[i][k];
            for (size_t j = 0; j < b_cols; ++j)
            {
                result[i][j] += val * b[k][j];
            }
        }
    }

    return result;
}

// Параллельное умножение
matrix multiply_parallel(const matrix &a, const matrix &b)
{
    assert(matrix_utils::columns(a) == matrix_utils::rows(b));

    const size_t a_rows = matrix_utils::rows(a);
    const size_t a_cols = matrix_utils::columns(a);
    const size_t b_cols = matrix_utils::columns(b);

    matrix result = matrix_utils::create(a_rows, b_cols);

#pragma omp parallel for num_threads(kNumThreads) schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(a_rows); ++i)
    {
        for (size_t k = 0; k < a_cols; ++k)
        {
            const int val = a[i][k];
            for (size_t j = 0; j < b_cols; ++j)
            {
                result[i][j] += val * b[k][j];
            }
        }
    }

    return result;
}

// Генерация матрицы с thread-safe random
matrix generate_matrix(size_t rows, size_t cols)
{
    matrix result = matrix_utils::create(rows, cols);

#pragma omp parallel num_threads(kNumThreads)
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_int_distribution<> dist(-100, 100);

#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(rows); ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result[i][j] = dist(gen);
            }
        }
    }

    return result;
}

int main()
{
    omp_set_num_threads(kNumThreads);

    // Генерация матриц
    Timer gen_timer;
    matrix a = generate_matrix(kMatrixSize, kMatrixSize);
    matrix b = generate_matrix(kMatrixSize, kMatrixSize);
    std::cout << "Generation time: " << gen_timer.elapsed() << " ms\n";

    // Последовательное умножение
    Timer serial_timer;
    matrix serial_result = multiply_serial(a, b);
    std::cout << "Serial multiplication: " << serial_timer.elapsed() << " ms\n";

    // Параллельное умножение
    Timer parallel_timer;
    matrix parallel_result = multiply_parallel(a, b);
    std::cout << "Parallel multiplication: " << parallel_timer.elapsed() << " ms\n";

    // Проверка корректности
    assert(matrix_utils::are_equal(serial_result, parallel_result));

    return 0;
}
