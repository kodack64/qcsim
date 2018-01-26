
#include <cstdio>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <chrono>
#include <bitset>
#include <iostream>
#include <complex>
#include <conio.h>
#include <vector>
#include <fstream>

using namespace std;

class MyCuda {
private:
	unsigned int _n;
	unsigned int _dim;
	complex<double> *stateOrg;
	complex<double> *stateNext;

public:
	MyCuda() :_n(0), _dim(0), stateOrg(0), stateNext(0)
	{
	}
	virtual ~MyCuda() {
	}

	void init(int numQubit) {
		_n = numQubit;
		_dim = 1 << numQubit;

		stateOrg = new complex<double>[_dim];
		stateNext = new complex<double>[_dim];
		for (unsigned int i = 0; i < _dim; i++) stateOrg[i] = 0;
		stateOrg[0] = 1;
	}


	void apply1QG(int gid, int target) {
		//X
		if (gid == 0) {
			int shift = 1 << target;
			for (int i = 0; i < _dim; i++) {
				stateNext[i] = stateOrg[i ^ shift];
			}
		}
		//Y
		else if (gid == 1) {
			int shift = 1 << target;
			for (int i = 0; i < _dim; i++) {
				int sign = 1 - ((i >> target) % 2) * 2;
				stateNext[i] = double(sign) * complex<double>(0, 1) * stateOrg[i^shift];
			}
		}
		// H
		else if (gid == 2) {
			int shift = 1 << target;
			for (int i = 0; i < _dim; i++) {
				if (i >> target) {
					stateNext[i] = (-stateOrg[i] + stateOrg[i^shift])*sqrt(0.5);
				}
				else {
					stateNext[i] = (stateOrg[i] + stateOrg[i^shift])*sqrt(0.5);
				}
			}
		}
		// T
		else if (gid == 3) {
			for (int i = 0; i < _dim; i++) {
				if ((i >> target) % 2) {
					stateNext[i] = complex<double>(sqrt(0.5), sqrt(0.5)) * stateOrg[i];
				}
				else {
					stateNext[i] = stateOrg[i];
				}
			}
		}
		// S
		else if (gid == 4) {
			for (int i = 0; i < _dim; i++) {
				if ((i >> target) % 2) {
					stateNext[i] = complex<double>(0,1) * stateOrg[i];
				}
				else {
					stateNext[i] = stateOrg[i];
				}
			}
		}
	}
	void apply2QG(int gid, int control, int target) {
		const int shift = 1 << target;
		for (int i = 0; i < _dim; i++) {
			if ((i >> control) % 2) {
				stateNext[i] = stateOrg[i^shift];
			}
			else {
				stateNext[i] = stateOrg[i];
			}
		}
	}

	void close() {
		delete[] stateOrg;
		delete[] stateNext;
	}
};

vector<__int64> randomCircuitOneshot(unsigned int n, unsigned int depth) {
	std::mt19937 mt(0);
	vector<__int64> times;

	MyCuda* mc = new MyCuda();
	mc->init(n);

	std::fstream ofs("cputime.txt", std::ios::app);
	ofs << n << " ";
	ofs.close();

	for (int d = 0; d < depth; d++) {
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < n; i++) {
			int r = mt() % 5;
			if (r != 4) {
				mc->apply1QG(r,i);
			}
			else {
				if (i + 1 < n) {
					mc->apply2QG(r, i, i + 1);
					i++;
				}
				else {
					mc->apply1QG(r, i);
				}
			}
		}
		__int64 time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
		std::cout << d << " " << time << std::endl;

		std::fstream ofsa("cputime.txt", std::ios::app);
		ofsa << time << " ";
		ofsa.close();
	}
	mc->close();
	std::fstream ofse("cputime.txt", std::ios::app);
	ofse << std::endl;
	ofse.close();

	return times;
}


int main(int argc, char** argv) {
	int n = 27;
	int depth = 100;
	if (argc > 1) {
		n = atoi(argv[1]);
		depth = atoi(argv[2]);
	}
	auto time = randomCircuitOneshot(n, depth);
	return 0;
}