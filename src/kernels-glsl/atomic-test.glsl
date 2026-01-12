#version 450 core

// A simple counter and a lock flag
layout(std430, binding = 0) buffer Data {
    uint counter;
    uint lock[];  // 0 = unlocked, 1 = locked
};

// Shared memory for workgroup-local data
shared uint local_sum;

void main() {
    // Initialize shared memory
    local_sum = 0u;

    // Each work-item does some local work
    uint my_value = gl_GlobalInvocationID.x * 2u;
    atomicAdd(local_sum, my_value);

    // Synchronize inside the workgroup
    barrier();

    // Only one work-item per workgroup will update the global counter
    if (gl_LocalInvocationIndex == 0u) {
        // Try to acquire the lock using atomic compare-and-swap
        uint expected = 0u;  // We expect the lock to be free
        uint desired  = 1u;  // We want to set it to locked

        // atomicCompSwap returns the old value of 'lock'
        uint poop = lock[0];
        uint old = atomicCompSwap(lock[0], expected, desired);

        // If old == 0, we successfully acquired the lock
        if (old == 0u) {
            // Critical section: safely update the global counter
            counter += local_sum;

            // Release the lock (set it back to 0)
            atomicExchange(lock[0], 0u);
        }
        // If old != 0, someone else holds the lock; we just skip the update
    }
}
