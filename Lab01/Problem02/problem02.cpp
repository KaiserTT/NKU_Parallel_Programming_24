#include <iostream>
#include <windows.h>
#include <cmath>
using namespace std;

#define LOOP 10
#define N 33554432

unsigned long long int a[N];


void init() {
    for (int i = 0; i < N; ++i)
        a[i] = i;
}

void ordinary(int n) {
    unsigned long long int sum = 0;
    for (int i = 0; i < n; ++i)
        sum += a[i];
}

void two_way(int n) {
    unsigned long long int sum = 0;
    unsigned long long int sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
}

void recursion(int n) {
    if (n == 1)
        return;
    else {
        for (int i = 0; i < n / 2; ++i)
            a[i] += a[n - 1 - i];
        recursion(n / 2);
    }
}

int main() {
    init();
    long double a1[LOOP], a2[LOOP], a3[LOOP];
    long long head, tail, freq;
    for (int i = 10; i <= 25; ++i) {
        for (int j = 0; j < LOOP; j++) {
            QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
            QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
            ordinary(pow(2, i));
            QueryPerformanceCounter((LARGE_INTEGER *)&tail) ;
            a1[j] = (tail - head) * 1000.0 / freq;
        }
        for (int j = 0; j < LOOP; j++) {
            QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
            QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
            two_way(pow(2, i));
            QueryPerformanceCounter((LARGE_INTEGER *)&tail) ;
            a2[j] = (tail - head) * 1000.0 / freq;
        }
        for (int j = 0; j < LOOP; j++) {
            QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
            QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
            recursion(pow(2, i));
            QueryPerformanceCounter((LARGE_INTEGER *)&tail) ;
            a3[j] = (tail - head) * 1000.0 / freq;
        }

        long double sum1 = 0, sum2 = 0, sum3 = 0;
        for (int j = 0; j < LOOP; j++) {
            sum1 += a1[j];
            sum2 += a2[j];
            sum3 += a3[j];
        }

        cout << sum1 / LOOP << ", ";
        cout << sum2 / LOOP << ", ";  
        cout << sum3 / LOOP << ", ";  
    }

    return 0;
}