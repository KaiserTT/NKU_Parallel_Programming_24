#include <iostream>
#include <windows.h>

using namespace std;

#define N 3000
#define LOOP 10

unsigned long long int v[N], m[N][N];

void init() {
    for (int i = 0; i < N; ++i) {
        v[i] = i;
        for (int j = 0; j < N; ++j) {
            m[i][j] = i;
        }
    }
}

void ordinary() {
    unsigned long long int sum[N] = {0};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            sum[i] += m[j][i] * v[j];
    }
}

void cache_optimize() {
    unsigned long long int sum[N] = {0};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            sum[j] += m[i][j] * v[i];
}

void unroll() {
    unsigned long long int sum[N] = {0};
    for (int i = 0; i < N; i += 10) {
        for (int j = 0; j < N; ++j) {
            sum[i + 0] += m[j][i + 0] * v[j];
            sum[i + 1] += m[j][i + 1] * v[j];
            sum[i + 2] += m[j][i + 2] * v[j];
            sum[i + 3] += m[j][i + 3] * v[j];
            sum[i + 4] += m[j][i + 4] * v[j];
            sum[i + 5] += m[j][i + 5] * v[j];
            sum[i + 6] += m[j][i + 6] * v[j];
            sum[i + 7] += m[j][i + 7] * v[j];
            sum[i + 8] += m[j][i + 8] * v[j];
            sum[i + 9] += m[j][i + 9] * v[j];
        }
    }
}

void cache_unroll() {
    unsigned long long int sum[N] = {0};
    for (int i = 0; i < N; i += 10) {
        for (int j = 0; j < N; ++j) {
            sum[j] += m[i + 0][j] * v[i + 0];
            sum[j] += m[i + 1][j] * v[i + 1];
            sum[j] += m[i + 2][j] * v[i + 2];
            sum[j] += m[i + 3][j] * v[i + 3];
            sum[j] += m[i + 4][j] * v[i + 4];
            sum[j] += m[i + 5][j] * v[i + 5];
            sum[j] += m[i + 6][j] * v[i + 6];
            sum[j] += m[i + 7][j] * v[i + 7];
            sum[j] += m[i + 8][j] * v[i + 8];
            sum[j] += m[i + 9][j] * v[i + 9];
        }
    }
}

int main() {
    init();
    double a1[LOOP], a2[LOOP], a3[LOOP], a4[LOOP];
    long long head, tail, freq;
    for (int i = 0; i < LOOP; i++) {
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
        QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
        ordinary();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail) ;
        a1[i] = (tail - head) * 1000.0 / freq;
    }
    for (int i = 0; i < LOOP; i++) {
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
        QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
        cache_optimize();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail) ;
        a2[i] = (tail - head) * 1000.0 / freq;
    }
    for (int i = 0; i < LOOP; i++) {
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
        QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
        unroll();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail) ;
        a3[i] = (tail - head) * 1000.0 / freq;
    }
    for (int i = 0; i < LOOP; i++) {
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
        QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
        cache_unroll();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail) ;
        a4[i] = (tail - head) * 1000.0 / freq;
    }
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (int i = 0; i < LOOP; i++) {
        sum1 += a1[i];
        sum2 += a2[i];
        sum3 += a3[i];
        sum4 += a4[i];
    }
        

    cout << sum1 / LOOP << "ms" << endl;
    cout << sum2 / LOOP << "ms" << endl;
    cout << sum3 / LOOP << "ms" << endl;
    cout << sum4 / LOOP << "ms" << endl;
}