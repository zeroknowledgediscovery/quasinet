#include <stdio.h>
#include <stdlib.h>

// Declare the dcor and wdcor functions from the library
double dcor(double *x, double *y, int n);
double wdcor(double *x, double *y, int n, double *w);

int main() {

  for(int run=0;run<5;run++)
    {
      int n = 5;

      double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
      double y[] = {2.0, 3.0, 4.0, 5.0, 6.0};
      double w[] = {0.1, 0.2, 0.3, 0.2, 0.2};

      double result_dcor = dcor(x, y, n);
      double result_wdcor = wdcor(x, y, n, w);

      printf("dcor: %f\n", result_dcor);
      printf("wdcor: %f\n", result_wdcor);
    }
  return 0;
}
