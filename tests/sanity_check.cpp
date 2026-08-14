#include <dlprim/random.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/ops/initialization.hpp>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <limits>
#include "test.hpp"

namespace dp = dlprim;

int main(int argc, char** argv)
{
	dp::Context ctx(argv[1]);

	
	dp::Tensor t(ctx.device(), dp::Shape(10) );
	std::vector<float> values(10);
	for (size_t i = 0; i < 10; i += 1)
		values[i] = (float)i;
	
	t.to_device(values.data());
	std::vector<float> expected(10, 0.0);
	t.to_host(expected.data());
	for (size_t i = 0; i < 10; i += 1)
	{
		std::cout << ((expected[i] == values[i]) ? "equal!" : "not!") << std::endl;
	}
	
	return 0;
	
}
