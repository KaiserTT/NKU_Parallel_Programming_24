#include <iostream>
#include <windows.h>

using namespace std;

#define MAX 2048
#define LOOP 10000

unsigned long long a[MAX] = {0};
unsigned long long sum = 0;

template<unsigned long long N, typename T>
struct ArraySum {
    static T sum(const T* array) {
        return array[N - 1] + ArraySum<N - 1, T>::sum(array);
    }
};

template<typename T>
struct ArraySum<0, T> {
    static T sum(const T* array) {
        return 0;
    }
};

void init() {
    for (int i = 0; i < MAX; ++i)
        a[i] = i;
}

void meta() {
    init();
    int total = ArraySum<MAX, unsigned long long>::sum(a);
}

void loop() {
    for (int i = 0; i < LOOP; ++i)
        meta();
}

int main() {
    double a1[LOOP];
    long long head, tail, freq;
    for (int i = 0; i < LOOP; i++) {
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
        QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
        meta();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail) ;
        a1[i] = (tail - head) * 1000.0 / freq;
    }

    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (int i = 0; i < LOOP; i++) {
        // cout << a1[i] << endl;
        sum1 += a1[i];
    }
    cout << sum1 / LOOP;

    return 0;
}