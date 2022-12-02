
#include "student.h"
#include <iostream>

int main()
{
    Student xiaoyun("xiaoyun", 28);
    float score[4] = {86, 59, 98, 100};
    xiaoyun.set_score(score, 4);
    return 0;
}