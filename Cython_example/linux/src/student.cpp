/*
 * @author     : yangyun
 * @date       : 2022-11-25
 * @description: student.cpp
 * @version    : 1.0
 */

#include <iostream>
#include "student.h"

Student::Student(char *name, int age)
    : m_name(name)
    , m_age(age)
{
    std::cout << "Creating a student: [name] " << m_name << " [age] " << m_age << std::endl;
}

void Student::set_score(float *score, int n)
{
    m_score = new float[n];
    
    for (int i = 0; i < n; i++)
    {
        m_score[i] = score[i];
    }
}

bool Student::delete_score()
{
    delete[] m_score;
    m_score = NULL;
    return true;
}


const uint32_t Student::get_age() const
{
    return m_age;
}

const float* Student::get_score() const
{
    return m_score;
}
