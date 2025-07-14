/*================================================================
*   Copyright (C) 2025 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : main.cpp
*   Author      : YunYang1994
*   Created date: 2025-07-14 16:23:24
*   Description :
*
*===============================================================*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include "serializer.h"


// 需要序列化的结构体
struct Person {
    int id;
    std::string name;
    std::vector<int> scores;

    template <typename Archive>
    void Serialize(Archive& archive) {
        archive & id;
        archive & scores;
        archive & name;
    }
};

// 保存到 bin 文件
void saveToFile(const Person& person, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "打开文件失败: " << path << std::endl;
        return;
    }
    Archive<std::ofstream> archive(ofs);
    archive << person;
    std::cout << "已保存到 " << path << std::endl;
}

// 从 bin 文件加载
Person loadFromFile(const std::string& path) {
    Person person;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "打开文件失败: " << path << std::endl;
        return person;
    }
    Archive<std::ifstream> archive(ifs);
    archive >> person;
    std::cout << "从 " << path << " 加载成功" << std::endl;
    return person;
}

// 主函数演示
int main() {
    Person p1;
    p1.id = 42;
    p1.name = "Alice";
    p1.scores = {90, 85, 88};

    // 保存
    saveToFile(p1, "person.bin");

    // 读取
    Person p2 = loadFromFile("person.bin");

    // 打印
    std::cout << "ID: " << p2.id << std::endl;
    std::cout << "Name: " << p2.name << std::endl;
    std::cout << "Scores: ";
    for (int score : p2.scores) std::cout << score << " ";
    std::cout << std::endl;

    return 0;
}

