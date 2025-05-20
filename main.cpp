#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

int main() {
    const int rows = 84;
    const int cols = 128;
    float* data = new float[rows * cols]; // or float data[rows * cols];

    std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
    std::ifstream infile(folder + "token_embedding_table.weight.txt");
    if (!infile) {
        std::cerr << "Could not open file.\n";
        return 1;
    }

    std::string line;
    int row = 0;
    while (std::getline(infile, line) && row < rows) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string val;
        int col = 0;
        while (iss >> val && col < cols) {
            data[row * cols + col] = std::stof(val);
            ++col;
        }
        ++row;
    }

    for (int i = 0; i < std::min(5, rows * cols); ++i)
        std::cout << data[i] << " ";
    std::cout << std::endl;

    delete[] data; 
    return 0;
}