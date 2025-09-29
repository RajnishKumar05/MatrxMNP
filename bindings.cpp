#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cblas.h>

namespace py = pybind11;

// Flatten 2D int matrix to 1D double
std::vector<double> flatten(const std::vector<std::vector<int>>& mat) {
    int N = mat.size();
    std::vector<double> flat(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            flat[i*N + j] = mat[i][j];
    return flat;
}

// Reshape 1D double array back to 2D int matrix
std::vector<std::vector<int>> reshape(const std::vector<double>& flat, int N) {
    std::vector<std::vector<int>> mat(N, std::vector<int>(N));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = static_cast<int>(flat[i*N + j]);
    return mat;
}

// Fast matrix multiplication using OpenBLAS
std::vector<std::vector<int>> matmul(const std::vector<std::vector<int>>& A,
                                     const std::vector<std::vector<int>>& B) {
    int N = A.size();
    std::vector<double> flatA = flatten(A);
    std::vector<double> flatB = flatten(B);
    std::vector<double> flatC(N * N, 0.0);

    // OpenBLAS multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, flatA.data(), N, flatB.data(), N, 0.0, flatC.data(), N);

    return reshape(flatC, N);
}

// Pybind11 module
PYBIND11_MODULE(cppmatrix, m) {
    m.doc() = "Fast matrix multiplication using OpenBLAS";
    m.def("matmul", &matmul, "Matrix multiplication using OpenBLAS",
          py::arg("A"), py::arg("B"));
}
