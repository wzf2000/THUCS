#include <iostream>

void bind_cpu() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(4, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
        perror("sched_setaffinity");
    }
}
