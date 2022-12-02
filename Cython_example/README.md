## Notion
Linux 使用的是 Student 类进行示例，Windows 使用的是 image 类读写处理

### linux
```cpp
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
```

### windows

```cpp
#ifdef _MSC_VER
#ifdef image_EXPORTS                          // 该宏在 windows 下会自动生成
#define IMAGE_EXPORT __declspec(dllexport)
#else
#define IMAGE_EXPORT __declspec(dllimport)
#endif
#else
#define IMAGE_EXPORT __attribute__((visibility("default")))
#endif

typedef IMAGE_EXPORT struct image{
    int w,h,c;
    unsigned char* data;
} image;

IMAGE_EXPORT void  save_image_stb(image im, const char *name, int png);
IMAGE_EXPORT image load_image_stb(char *filename, int channels);
IMAGE_EXPORT image Resize(unsigned char* data, int src_w, int src_h, int src_c, int dst_w, int dst_h);
```
