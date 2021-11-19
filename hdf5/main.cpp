
#include "H5Cpp.h"
#include <fstream>

class HFiveFile {
public:
	HFiveFile (const char* file_name) {
		file = H5::H5File(file_name, H5F_ACC_TRUNC);
	}

	void H5FileWriteFloat32(const float* data_ptr, const char* dataset_name, size_t* shape, int dim) {
		auto h_shape = static_cast<hsize_t*> (shape);
		H5::DataSpace dataspace(dim, h_shape);

		H5::IntType datatype(H5::PredType::NATIVE_FLOAT);
		datatype.setOrder(H5T_ORDER_LE);

		H5::DataSet dataset = file.createDataSet(dataset_name, datatype, dataspace);
		dataset.write(data_ptr, H5::PredType::NATIVE_FLOAT);
	}

private:
	H5::H5File file;
};


static float rotation[] = {1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                           9.95594144e-01, -9.35074911e-02,  8.96214917e-02,
                           9.76112187e-01, -1.16945654e-01, -7.56589174e-02};


int main () {
    // method 1
	HFiveFile H5write("rotation_method1.h5");
	size_t shape[] = {3,3};
	H5write.H5FileWriteFloat32(rotation, "rotation", shape, 2);

    // method 2
    std::fstream fout("rotation_method2.dat", std::ios::out | std::ios::binary);
    fout.write((char*)rotation, sizeof(float) * 3 * 3);
}


