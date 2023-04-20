#include <math.h>
#include <stddef.h>
#include <omp.h>

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

double avg_jsd(double* V1_list[], double* V2_list[], size_t list_length, size_t length) {
    double sum_jsd = 0.0;
    for (size_t i = 0; i < list_length; i++) {
        sum_jsd += jsd(V1_list[i], V2_list[i], length);
    }
    return sum_jsd / list_length;
}


void fill_matrix(double* matrix,
		 double** V1_list,
		 double** V2_list,
		 size_t num_seqs1,
		 size_t num_seqs2,
		 size_t length) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < num_seqs1; i++) {
        for (size_t j = i + 1; j < num_seqs2; j++) {
            double avg_jsd_value = avg_jsd(V1_list[i], V2_list[j], length, length);
            matrix[i * num_seqs2 + j] = avg_jsd_value;
            matrix[j * num_seqs1 + i] = avg_jsd_value;
        }
    }
}
