import itertools
import os
import argparse
from enum import Enum

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

ALL_DTYPES = []
for name in DTypeEnum._member_names_:
	ALL_DTYPES.append(DTypeEnum.__getattribute__(DTypeEnum, name))

ALL_FLOATS = []
for name in DTypeEnum._member_names_:
	if (not "DTYPE_I" in name) and (not "DTYPE_U" in name):
		ALL_FLOATS.append(DTypeEnum.__getattribute__(DTypeEnum, name))

ALL_INTS = []
for name in DTypeEnum._member_names_:
	if ("DTYPE_I" in name) or ("DTYPE_U" in name):
		ALL_INTS.append(DTypeEnum.__getattribute__(DTypeEnum, name))

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
			defs.append("ENABLE_BFLOAT16_ARITHMETIC")
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
		elif dt in [DTypeEnum.DTYPE_U64, DTypeEnum.DTYPE_I64]:
			defs.append("ENABLE_INT64_ARITHMETIC")
	return set(defs)

def get_defs_from_dtype_spec_permutation(spec_permutation):
	dtypes = []
	for item in spec_permutation:
		dtypes.append(item["type"])
	assert len(dtypes) > 0
	dtypes = set(dtypes)
	return get_defs_from_dtypes(dtypes)

def generate_spv_filename(file, dtype_specs):
	dtype_str = []
	for s in dtype_specs:
		dtype_str.append(f"D{s['type'].value}")
	dtype_str = "_".join(dtype_str)
	base = os.path.splitext(file)[0]
	return f"{base}_{dtype_str}.spv"

def compile_glsl(path, output_dir, dtype_specs):
	defs = get_defs_from_dtype_spec_permutation(item)
	dflags = []
	for d in defs:
		dflags.append(f"-D{d}=1")
	for spec in dtype_specs:
		dflags.append(f"-D{spec['name']}={spec['type'].value}")
	dflag_str = " ".join(dflags)
	fn = generate_spv_filename(os.path.basename(path), dtype_specs)
	out_path = os.path.join(output_dir, fn)
	return_code = os.system(f"glslc -fshader-stage=compute {path} {dflag_str} -Os -o {out_path}")
	if return_code > 0:
		raise RuntimeError(f"compilation failed for {path} with flags: {dflag_str}")
	
def get_dtype_specs(path):
	# This function gets the compatible dtype specifications (preprocessor definition + dtypes to be used)
	dtype_specs = []
	with open(path, "r") as f:
		src = f.read()
		for l in src.split("\n"):
			if "@dtype" in l:
				args = l.split("@dtype ")[1].split(" ")
				spec = {"name": args[0], "types": []}
				dtype_str_args = args[1:]
				for dts in dtype_str_args:
					if dts == "all":
						for name in DTypeEnum._member_names_:
							spec["types"].append(DTypeEnum.__getattribute__(DTypeEnum, name))
					elif dts == "floats":
						spec["types"] = ALL_FLOATS
					else:
						spec["types"].append(DTypeEnum.__getattribute__(DTypeEnum, dts))
				dtype_specs.append(spec)
	return dtype_specs
	
def iter_compatible_dtypes(dtype_specs):
	for p in itertools.permutations(ALL_DTYPES, len(dtype_specs)):
		all_compatible = True
		for i in range(len(dtype_specs)):
			if not p[i] in dtype_specs[i]["types"]:
				all_compatible = False
		if all_compatible:
			spec_permutation = []
			for i in range(len(dtype_specs)):
				spec_permutation.append(
					{"name": dtype_specs[i]["name"], "type": p[i]})
			yield spec_permutation

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
			if not os.path.isdir(path) and os.path.splitext(path)[1] == ".glsl":
				dtype_specs = get_dtype_specs(path)
				for item in iter_compatible_dtypes(dtype_specs):
					compile_glsl(path, args.o, item)
				
	
