# *ndarray*
Multi-dimensional array library for C++

## In a nutshell

    #include "ndarray/ndarray.h"
    using namespace ndarray;
    
    int main(int argc, char* argv[])
    {
        // construct an array of integers {0, 1, 2, ..., 99}, and reshape it to 10x10.
        auto a = reshape<2>(range(100), {10, 10});

        // copy the first row to the first column
        a(All, 0) = a(0);

        // flatten the first 5 rows
        auto b = flatten(a(span(5)));
        
        return 0;
    }
    
