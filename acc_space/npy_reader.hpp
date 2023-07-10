#ifndef NPY_READER_HPP
#define NPY_READER_HPP

#include <iostream>
#include <fstream>
#include <vector>

inline bool isLittleEndian() {
    int num = 1;
    return (*(char*)&num == 1);
}

// Read a tuple file stored in a .npy file
inline std::vector<double> readNpyFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        return {};
    }

    // Read the magic string and version number
    char header[12];
    file.read(header, 12);
    std::string magicString(header, 6);
    std::cout << "Magic string: " << magicString << std::endl;

    // Determine the endianness of the file
    bool littleEndian = (header[8] == '<');
    bool machineIsLittleEndian = isLittleEndian();
    if (littleEndian != machineIsLittleEndian) {
        std::cout << "Mismatch in endianness. Swapping byte order." << std::endl;
    }

    // Read the data type
    uint8_t dataType[2];
    file.read(reinterpret_cast<char*>(dataType), 2);
    uint16_t dataTypeCode = (dataType[1] << 8) | dataType[0];

    std::cout << "Data type code (hex): " << std::hex << dataTypeCode << std::dec << std::endl;
    std::cout << "Data type code (bytes): " << std::hex << static_cast<int>(dataType[0]) << " " << static_cast<int>(dataType[1]) << std::dec << std::endl;


    std::string dataTypeStr;
    switch (dataTypeCode) {
    case 0x09:
        dataTypeStr = "double";
        break;
    default:
        std::cout << "Unsupported data type: " << std::hex << dataTypeCode << std::dec << std::endl;
        file.close();
        return {};
    }

    std::cout << "Data type: " << dataTypeStr << std::endl;

    // Read the shape of the array
    uint16_t shapeLen;
    file.read(reinterpret_cast<char*>(&shapeLen), 2);
    std::vector<uint32_t> shape(shapeLen);
    file.read(reinterpret_cast<char*>(shape.data()), shapeLen * sizeof(uint32_t));

    // Read the tuple data
    uint32_t tupleSize = shape[0];
    std::vector<double> tuple(tupleSize);
    file.read(reinterpret_cast<char*>(tuple.data()), tupleSize * sizeof(double));

    file.close();

    // Swap byte order if endianness mismatch
    if (littleEndian != machineIsLittleEndian) {
        for (size_t i = 0; i < tupleSize; ++i) {
            uint64_t* dataPtr = reinterpret_cast<uint64_t*>(&tuple[i]);
            *dataPtr = ((*dataPtr & 0xFF00000000000000ull) >> 56) |
                ((*dataPtr & 0x00FF000000000000ull) >> 40) |
                ((*dataPtr & 0x0000FF0000000000ull) >> 24) |
                ((*dataPtr & 0x000000FF00000000ull) >> 8) |
                ((*dataPtr & 0x00000000FF000000ull) << 8) |
                ((*dataPtr & 0x0000000000FF0000ull) << 24) |
                ((*dataPtr & 0x000000000000FF00ull) << 40) |
                ((*dataPtr & 0x00000000000000FFull) << 56);
        }
    }

    return tuple;
}


#endif  // NPY_READER_HPP