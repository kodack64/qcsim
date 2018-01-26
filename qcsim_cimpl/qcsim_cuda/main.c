
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "simulator.h"
int main(int argc,char** argv) {
	if(argc==1) simulateConsoleStream();
	else if(argc==2){
		FILE* inStream = fopen(argv[1],"r");
		simulateFileStream(inStream,stdout);
		fclose(inStream);
	}
	else {
		FILE* inStream = fopen(argv[1], "r");
		FILE* outStream = fopen(argv[2], "w");
		simulateFileStream(inStream, outStream);
		fclose(outStream);
		fclose(inStream);
	}
	return 0;
}
