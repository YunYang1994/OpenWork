/*
 * @author     : yangyun
 * @date       : 2022-11-25
 * @description: student.h
 * @version    : 1.0
 */
#include <stdint.h>

#ifdef _MSC_VER
#ifdef STUDENT_EXPORTS
#define STUDENT_EXPORT __declspec(dllexport)
#else
#define STUDENT_EXPORT __declspec(dllimport)
#endif // STUDENT_EXPORTS
#else
#define STUDENT_EXPORT __attribute__((visibility("default")))
#endif

class STUDENT_EXPORT Student
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
