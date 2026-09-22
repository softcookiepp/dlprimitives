import itertools
import os
import argparse
from enum import Enum

if False:
	DTYPE_ENUM = [
		0,
		1,
		2,
		#3, BF16, not implemented
		4,
		5,
		#6, MXFP4, not implemented
		7,
		8,
		9,
		10,
		11,
		12,
		13,
		14
		#15, FP8, not implemented
		#16 same
	]

class DTypeEnum(Enum):
	DTYPE_F16 = 0
	DTYPE_F32 = 1
	DTYPE_F64 = 2
	DTYPE_BF16 = 3
	#DTYPE_TQ1 = 4
	#DTYPE_TQ2 = 5
	#DTYPE_MXFP4 6 // MXFP4 (1 block)
	DTYPE_U8 = 7
	DTYPE_U16 = 8
	DTYPE_U32 = 9
	DTYPE_U64 = 10
	DTYPE_I8 = 11
	DTYPE_I16 = 12
	DTYPE_I32 = 13
	DTYPE_I64 = 14
	#DTYPE_FP8E5M2 15
	#DTYPE_FP8E4M3 16
	#DTYPE_UNKNOWN 17

def get_defs_from_dtypes(dtypes):
	# This function will provide the -D flags necessary to enable a given set of dtypes
	dtypes = set(dtypes)
	defs = []
	for dt in dtypes:
		if dt == DTypeEnum.DTYPE_F16:
			defs.append("ENABLE_16BIT_STORAGE")
			defs.append("ENABLE_FLOAT16_ARITHMETIC")
		elif dt == DTypeEnum.DTYPE_BF16:
			defs.append("ENABLE_16BIT_STORAGE")
			
		elif dt == DTypeEnum.DTYPE_F64:
			defs.append("ENABLE_FLOAT64_ARITHMETIC")
		elif dt in [DTypeEnum.DTYPE_U8, DTypeEnum.DTYPE_I8]:
			defs.append("ENABLE_8BIT_STORAGE")
			defs.append("ENABLE_INT8_ARITHMETIC")
		elif dt in [DTypeEnum.DTYPE_U16, DTypeEnum.DTYPE_I16]:
			defs.append("ENABLE_16BIT_STORAGE")
			defs.append("ENABLE_INT16_ARITHMETIC")


def compile_glsl(path, output_dir, dtypes):
	raise NotImplementedError
	
def get_dtype_names(path):
	raise NotImplementedError

if __name__ == "__main__":
	default_src_path = os.path.dirname(__file__) # the script occupies the same folder as all the files
	default_bin_path = os.path.join(default_src_path, "SPIRV")
	parser = argparse.ArgumentParser()
	parser.add_argument('-I', action = 'append', default = [default_src_path])
	parser.add_argument('-o', default = default_bin_path)
	args = parser.parse_args()
	print(args.I)
	print(args.o)
	if not os.path.exists(args.o):
		os.mkdir(args.o)
	for d in args.I:
		d = os.path.abspath(d)
		for fn in os.listdir(d):
			path = os.path.join(d, fn)
			if not os.path.isdir(path):
				dtypes = []
				
				input(path)
	
