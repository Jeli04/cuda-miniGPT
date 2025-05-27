#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

float* loadMatrix(int rows, int cols, std::string& source){
    float* data = new float[rows * cols]; // or float data[rows * cols];
  
    std::ifstream infile(source);
    if (!infile) {
        std::cerr << "Could not open file.\n";
        exit(1);
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
  
    return data;
}

void dumpMatrix(float* matrix, int rows, int cols, const std::string& destination) {
    std::ofstream outfile(destination);
    if (!outfile.is_open()) {
        std::cerr << "Could not open file for writing: " << destination << std::endl;
        return;
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            outfile << matrix[r * cols + c];
            if (c < cols - 1)
                outfile << " ";
        }
        outfile << "\n";
    }
    outfile.close();
}


void printMatrix(float* matrix, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%.4f ", matrix[r * cols + c]);
        }
        printf("\n");
    }
}