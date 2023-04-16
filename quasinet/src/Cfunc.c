#include <math.h>
#include <stddef.h>

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
