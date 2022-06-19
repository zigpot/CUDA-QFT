#include "quantum_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// The definition of epsilon from libquantum's defs.h
#define epsilon 1e-6

// Some functions from libquantum's complex.h
static inline float
quantum_real(COMPLEX_FLOAT a)
{
  float *p = (float *) &a;
  return p[0];
}
static inline float
quantum_imag(COMPLEX_FLOAT a)
{
  float *p = (float *) &a;
  return p[1];
}

float norm2(cuDoubleComplex v){
	float result = 0.0;
	result += pow(v.x, 2);
	result += pow(v.y, 2);
	return result;
}

/*
 * Expand a quantum_reg into a vector of 2^qr.width states.
 * (*v) will be freed and reallocated.
 */
void qutil_expand_qr(const quantum_reg& qr, cuDoubleComplex **v)
{
    //printf("expand_qureg(width=%d)...", qr.width);
    //fflush(stdout);

    // Reisze and zero the vector.
    qutil_new_qvec(qr.width, v);

    // Copy in the non-zero states.
    int i;
    for (i = 0; i < qr.size; i++) {
        unsigned long long s = qr.state[i];
        (*v)[s].x = quantum_real(qr.amplitude[i]);
        (*v)[s].y = quantum_imag(qr.amplitude[i]);
    }

    //printf("done.\n");
    //fflush(stdout);
}

/*
 * Collapse a vector of 2^width states
 */
quantum_reg qutil_collapse_qr(int width, const cuDoubleComplex *v)
{
    //printf("collapse_qureg(width=%d)...", width);
    //fflush(stdout);

    unsigned long long N = (1ull << width);

    // First count the number of non-tiny states.
    float limit = (1.0 / (1ull << width)) * epsilon;

    unsigned long long i, j, count = 0;
    for (i = 0; i < N; i++) {
        if ((v[i].x*v[i].x + v[i].y*v[i].y) >= limit) {
            count++;
        }
    }

    // Allocate the qureg data structures.
    quantum_reg qr;
    qr.width = width;
    qr.size = count;
    qr.hashw = 0;
    qr.hash = NULL;
	qr.amplitude = (__complex__ float* )calloc(count, sizeof(__complex__ float*));
	qr.state = (long long unsigned int* )calloc(count, sizeof(long long unsigned int));
    if (!qr.amplitude || !qr.state) {
        fprintf(stderr, "Error allocating memory for %llu states in qureg.\n", count);
        exit(1);
    }

    j = 0;
    for (i = 0; i < N; i++) {
        if ((v[i].x*v[i].x + v[i].y*v[i].y) >= limit) {
            // Store this one.
            qr.state[j] = i;
            COMPLEX_FLOAT a = v[i].x + v[i].y*1i;
            qr.amplitude[j] = a;
            j++;
        }
    }


    // Note: libquantum doesn't expose a way to reconstruct the hash table.
    // I don't know what the hash table is used for. This may cause libquantum calls to fail.

    //printf("done.\n");
    //fflush(stdout);

    return qr;
}



/*
 * Reads a qureg from "filename". Returns the number of qbits (qr.width).
 */
int qutil_read_qr(const char *filename, cuDoubleComplex **v)
{
    unsigned long long i;
    FILE *fs = fopen(filename, "rb");
    if (!fs) {
        fprintf(stderr, "Could not open '%s' for reading.\n", filename);
        exit(1);
    }
    // First read the number of bits and states.
    int width, reg_size;
    if (!fread(&width, sizeof(int), 1, fs)) {
        fprintf(stderr, "Error reading number of qubits from '%s'.\n", filename);
        exit(1);
    }
    if (!fread(&reg_size, sizeof(int), 1, fs)) {
        fprintf(stderr, "Error reading number of states from '%s'.\n", filename);
        exit(1);
    }

    // Reisze and zero the vector.
    qutil_new_qvec(width, v);

    // Read in the states.
    for (i = 0; i < (unsigned long long )reg_size; i++) {
        unsigned long long st;
        float tmp[2];
        if (fread(&st, sizeof(unsigned long long), 1, fs) != 1) {
            fprintf(stderr, "Error reading %llu'th state from '%s'.\n", i, filename);
            exit(1);
        }
        if (fread(tmp, sizeof(float), 2, fs) != 2) {
            fprintf(stderr, "Error reading element %llu from '%s'.\n", i, filename);
            exit(1);
        }
        (*v)[st].x = tmp[0];
        (*v)[st].x = tmp[1];
    }
    return width;
}


/*
 * Allocate or resize vector *v, then zero the vector.
 * Exits the program with an error message if allocation fails.
 */
void qutil_new_qvec(int width, cuDoubleComplex **v)
{
    unsigned long long N = (1ul << width);
    if (*v) {
        *v = (cuDoubleComplex*)realloc(*v, N*sizeof(cuDoubleComplex));
    } else {
        *v = (cuDoubleComplex*)malloc(N*sizeof(cuDoubleComplex));
    }
    if (!*v) {
        fprintf(stderr, "Error allocating memory for vector (size %llu).\n", N);
        exit(1);
    }

    memset(*v, 0, N*sizeof(cuDoubleComplex));
}


/*
 * De-allocate the vector. It is safe to call this multiple times.
 */
void qutil_destroy_qvec(cuDoubleComplex **v)
{
    if (!*v) {
        // Do nothing; already freed.
    } // else...
    free(*v);
    *v = NULL;
}


/*
 * Copy vector vsrc into *vdst, allocating *vdst if necessary.
 * (The parameter order is similar to memcpy().)
 */
void qutil_copy_qvec(cuDoubleComplex **vdst, const cuDoubleComplex *vsrc, int width)
{
    qutil_new_qvec(width, vdst);
    unsigned long N = (1ul << width);
    memcpy(*vdst, vsrc, sizeof(cuDoubleComplex)*N);
}


/*
 * Read vector *v from a file, allocating it if necessary.
 */
int qutil_read_qvec(const char *filename, cuDoubleComplex **v)
{
    FILE *fs = fopen(filename, "rb");
    if (!fs) {
        fprintf(stderr, "Error opening '%s' for reading.\n", filename);
        exit(1);
    }
    int width = 0;
    if (fread(&width, sizeof(int), 1, fs) != 1) {
        fprintf(stderr, "Error reading number of qubits to '%s'.\n", filename);
    }

    qutil_new_qvec(width, v);

    unsigned long long N = (1ull << width);
    if (fread(*v, sizeof(float)*2, N, fs) != 2) {
        fprintf(stderr, "Error reading states from '%s'.\n", filename);
        exit(1);
    }

    fclose(fs);
    return width;
}


/*
 * Write vector v to a file.
 */
void qutil_write_qvec(const char *filename, int width, const cuDoubleComplex *v)
{
    FILE *fs = fopen(filename, "wb");
    if (!fs) {
        fprintf(stderr, "Error opening '%s' for writing.\n", filename);
        exit(1);
    }
    if (fwrite(&width, sizeof(int), 1, fs) != 1) {
        fprintf(stderr, "Error writing number of qubits to '%s'.\n", filename);
    }

    unsigned long long N = (1ull << width);
    if (fwrite(v, sizeof(float)*2, N, fs) != N) {
        fprintf(stderr, "Error writing states to '%s'.\n", filename);
        exit(1);
    }
    fclose(fs);
}


/*
 * Prints out vector v.
 */
void qutil_print_qvec(int width, const cuDoubleComplex *v)
{
    unsigned long N = (1ul << width);
    unsigned long i;
    for (i = 0; i<N; i++) {
        if (norm2(v[i]) >= 1e-6) {
            printf("(% f %+fi)|%li> (|", v[i].x, v[i].y, i);
            for(int j = width-1; j>=0; j--) {
                printf("%i", ((1ul << j) & i) != 0);
            }
            printf(">)\n");
        }
    }
}


/*
 * Calculates the L2-norm (sqrt(sum of squared norms of elements)) of two vectors.
 */
double qutil_l2norm_qvecs(int width, const cuDoubleComplex *v1, const cuDoubleComplex *v2)
{
    double L = 0.0;
    unsigned long N = (1ul << width);
    for (unsigned long i = 0; i < N; i++) {
        L += double(norm2(cuCsub(v1[i], v2[i])));
    }
    L = sqrt(L);
    return L;
}
