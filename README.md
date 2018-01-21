# *ndarray*
Multi-dimensional array library for C++

## In a nutshell

    #include "ndarray/ndarray.h"
    
    int main(int argc, char* argv[])
    {
        ndarray::array<int, 3> arr({10, 10, 2});        // create a 3-dimensional 10x10x2 array
        
        auto first_view  = arr.part_view(All, All, 0);  // create a 10x10 view with first  elements in the pairs
        auto second_view = arr.part_view(All, All, 1);  // create a 10x10 view with second elements in the pairs
        
        first_view. traverse([](auto& x) { x = 3; });   // set all elements in first_view to be 3
        second_view.traverse([](auto& x) { x = 5; });   // set all elements in first_view to be 5
        
        return 0;
    }
    
