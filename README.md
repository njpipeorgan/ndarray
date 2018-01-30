# *ndarray*
Multi-dimensional array library for C++

## In a nutshell

    #include "ndarray/ndarray.h"
    using namespace ndarray;
    
    int main(int argc, char* argv[])
    {
        // construct an array of integers {0, 1, 2, ..., 99}, and reshape it to 10x10.
        auto a1 = reshape<2>(range(100), {10, 10});
        
        // construct an array from the first column of a1
        auto a2 = flatten(make_array(a1(All, 1)));

        // copy the third column of a1 to the first column
        a1(All, 1) = a1(All, 3);
        
        return 0;
    }
    
