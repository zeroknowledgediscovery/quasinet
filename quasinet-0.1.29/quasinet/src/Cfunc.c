#include <math.h>
#include <stddef.h>
#include <omp.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>


double kld(double* P, double* Q, size_t length) {
  double divergence = 0.0;
  for (size_t i = 0; i < length; i++) {
    if (P[i] > 0 && Q[i] > 0) {
      divergence += P[i] * log2(P[i] / Q[i]);
    }
  }
  return divergence;
}


double jsd(double* V1, double* V2, size_t length) {
  // Calculate the Jensen-Shannon divergence
  double M[length];
  for (size_t i = 0; i < length; i++) {
    M[i] = (V1[i] + V2[i]) * 0.5;
  }
    
  double jsd = (kld(V1, M, length) + kld(V2, M, length)) * 0.5;
    
  return jsd;
}


double avg_jsd(double* V1_list[],
	       double* V2_list[],
	       int list_length,
	       size_t length[]) {
  double sum_jsd = 0.0;
  for (size_t i = 0; i < list_length; i++)
    {
      double jsd_ = jsd(V1_list[i],
			V2_list[i],
			length[i]);
      if (jsd_ > 0.0000000001)
	sum_jsd += sqrt(jsd_);
    }
    
  return sum_jsd / list_length;
}


int **find_common_indices(char **arr1, size_t n1, char **arr2, size_t n2, int *common_count) {
  int **common_indices = (int **)malloc(sizeof(int *) * 2);
  common_indices[0] = (int *)malloc(sizeof(int) * (n1 < n2 ? n1 : n2));
  common_indices[1] = (int *)malloc(sizeof(int) * (n1 < n2 ? n1 : n2));
    
  *common_count = 0;

  for (size_t i = 0; i < n1; i++) {
    for (size_t j = 0; j < n2; j++) {
      if (strcmp(arr1[i], arr2[j]) == 0) {
	common_indices[0][*common_count] = i;
	common_indices[1][*common_count] = j;
	(*common_count)++;
      }
    }
  }
  return common_indices;
}


double avg_jsd_(double* V1_list[],
		char** key1_list[],
		size_t key1_length[],
		double* V2_list[],
		char** key2_list[],
		size_t key2_length[],
		int list_length) {
  double sum_jsd = 0.0;
    
  for (size_t i = 0; i < list_length; i++)
    {
      double* V1 = V1_list[i];
      double* V2 = V2_list[i];
      int common_count;
      int **common_indices = find_common_indices(key1_list[i],key1_length[i],
						 key2_list[i],key2_length[i],
						 &common_count);

      double *V1_list_ = (double *)malloc(sizeof(double ) * common_count);
      double *V2_list_ = (double *)malloc(sizeof(double ) * common_count);
	
      for (int k=0;k<common_count;k++)
	{
	  V1_list_[k]=V1_list[i][common_indices[0][k]];
	  V2_list_[k]=V2_list[i][common_indices[1][k]];
	}
       
      double jsd_ = jsd(V1_list_,
			V2_list_,
			common_count);
      if (jsd_ > 0.0000000001)
	sum_jsd += sqrt(jsd_);

      free(V1_list_);
      free(V2_list_);

      // Assuming find_common_indices() allocates memory for common_indices
      // Free the memory allocated by find_common_indices()
      free(common_indices[0]);
      free(common_indices[1]);
      free(common_indices);
    }
  return sum_jsd / list_length;
}

void fill_matrix_symmetric(double* matrix,
			   double** V1_list[],
			   char*** key1_list[],
			   size_t* key1_length[],
			   int list_length, // num features
			   int num_seqs1
			   )
{
#pragma omp parallel for collapse(2)
  for (int i = 0; i < num_seqs1; ++i)
    for (int j = i + 1; j < num_seqs1; ++j)
      {	
	double avg_jsd_value = avg_jsd_(V1_list[i],
					key1_list[i],
					key1_length[i],
					V1_list[j],
					key1_list[j],
					key1_length[j],
					list_length);
	matrix[i * num_seqs1 + j] = avg_jsd_value;
	matrix[j * num_seqs1 + i] = avg_jsd_value;
      }
}




void fill_matrix(double* matrix,
		 double** V1_list[],
		 char*** key1_list[],
		 size_t* key1_length[],
		 double** V2_list[],
		 char*** key2_list[],
		 size_t* key2_length[],
		 int list_length, // this is the num of features
		 int num_seqs1,
		 int num_seqs2
		 )
{
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < num_seqs1; ++i)
    for (int j = 0; j < num_seqs2; ++j)
      matrix[i * num_seqs2 + j] = avg_jsd_(V1_list[i],
					   key1_list[i],
					   key1_length[i], 
					   V2_list[j],
					   key2_list[j],
					   key2_length[j],
					   list_length);
}


