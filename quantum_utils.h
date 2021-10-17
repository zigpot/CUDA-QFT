#ifndef QUANTUM_UTILS_H_INCLUDED
#define QUANTUM_UTILS_H_INCLUDED


// This file contains a collection of utilities for manipulating quantum_reg
// data structures from libquantum.


#include <cuComplex.h>

extern "C" {
    #include <quantum.h>
};

void qutil_expand_qr(const quantum_reg& qr, cuDoubleComplex **v);
quantum_reg qutil_collapse_qr(int width, const cuDoubleComplex *v);

int qutil_read_qr(const char *filename, cuDoubleComplex **v);

void qutil_new_qvec(int width, cuDoubleComplex **v);
void qutil_destroy_qvec(cuDoubleComplex **v);
void qutil_copy_qvec(cuDoubleComplex **vdst, const cuDoubleComplex *vsrc, int width);
int qutil_read_qvec(const char *filename, cuDoubleComplex **v);
void qutil_write_qvec(const char *filename, int width, const cuDoubleComplex *v);
void qutil_print_qvec(int width, const cuDoubleComplex *v);

double qutil_l2norm_qvecs(int width, const cuDoubleComplex *v1, const cuDoubleComplex *v2);

#endif // QUANTUM_UTILS_H_INCLUDED
