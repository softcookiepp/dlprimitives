// file specifying the specification constants required for variable local size
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(constant_id = 0) const uint localSizeX = 1;
layout(constant_id = 1) const uint localSizeY = 1;
layout(constant_id = 2) const uint localSizeZ = 1;
