// Independent double-precision CPU oracle for the rank-3 polynomial gradients.
#include <cmath>
#include <complex>

using Complex = std::complex<double>;

#ifdef _WIN32
#define AFM_EXPORT __declspec(dllexport)
#else
#define AFM_EXPORT
#endif

extern "C" AFM_EXPORT void rank3(int count, const double* xyz,
                                          const double* moment, double* out) {
  Complex sum[7]{};
  for (int i = 0; i < count; ++i) {
    const double x = xyz[3*i], y = xyz[3*i+1], z = xyz[3*i+2];
    const Complex u(x, -y);
    const double v = 4*z*z-x*x-y*y;
    const Complex g[4][3] = {
      {-3*z*x, -3*z*y, 3*z*z-1.5*(x*x+y*y)},
      {-std::sqrt(3.)/4.*(v-2.*x*u),
       -std::sqrt(3.)/4.*(Complex(0,-1)*v-2.*y*u),
       -std::sqrt(3.)/4.*8.*z*u},
      {std::sqrt(15./8)*2.*z*u, std::sqrt(15./8)*Complex(0,-2)*z*u,
       std::sqrt(15./8)*u*u},
      {-std::sqrt(5.)/4.*3.*u*u, -std::sqrt(5.)/4.*Complex(0,-3)*u*u, 0.}
    };
    for (int q=0; q<4; ++q) {
      Complex s{};
      for (int a=0; a<3; ++a) s += g[q][a]*moment[3*i+a];
      sum[3+q] += s;
      if (q) sum[3-q] += (q%2 ? -1. : 1.)*std::conj(s);
    }
  }
  for (int q=0; q<7; ++q) { out[2*q]=sum[q].real(); out[2*q+1]=sum[q].imag(); }
}
