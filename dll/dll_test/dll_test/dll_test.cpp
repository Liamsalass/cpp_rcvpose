// dll_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "rcvpose.h"

int main()
{
    Options opts;

    RCVpose test(opts);

    test.summary();
 
}

