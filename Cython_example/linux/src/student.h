/*
 * Copyright 2022 YunYang1994 All Rights Reserved. 
 * @Author: YunYang1994
 * @FilePath: student.h
 * @Date: 2022-12-01 16:15:20
 */

#include <stdint.h>

class Student
{
public:
    Student(char* name, int age);

    void set_score(float* score, int n);
    bool delete_score();

    const uint32_t get_age() const;
    const float* get_score() const;   

private:
    char* m_name;
    int m_age;
    float* m_score;
};
