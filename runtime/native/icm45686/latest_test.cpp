#include "ipc.hpp"
#include "policy_imu_client.hpp"
#include <iostream>

using sprite_policy::ImuSnapshot;
using sprite_policy::NativeImuClient;
using sprite_policy::monotonic_ns;

static void require(bool ok) {
    if (!ok) throw std::runtime_error("direct IMU snapshot test failed");
}

static void expect_error(NativeImuClient& client, const char* reason) {
    ImuSnapshot s;
    try { client.snapshot(s); }
    catch (const std::runtime_error& e) {
        require(std::string(e.what()) == reason);
        return;
    }
    throw std::runtime_error("invalid snapshot accepted");
}

int main(int argc, char** argv) {
    ImuSnapshot s;
    if (argc == 2) {
        // Used by the vcan test with a separate, live 800 Hz producer.
        NativeImuClient client(std::stoi(argv[1]));
        require(sprite_schedule::tids().size() == 1);
        std::uint64_t previous = 0;
        for (int i = 0; i < 20; ++i) {
            require(client.snapshot(s) && s.sequence > previous);
            require(std::abs(s.values[0] + 9.80665) < 1e-10);
            previous = s.sequence;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return 0;
    }
    int fd = memfd_create("latest-imu-test", MFD_ALLOW_SEALING | MFD_CLOEXEC);
    require(fd >= 0);
    const std::string descriptor = std::to_string(fd);
    IcmServer producer(nullptr, descriptor.c_str());
    NativeImuClient client(fd);
    close(fd);
    require(sprite_schedule::tids().size() == 1 && !client.snapshot(s));
    const auto publish = [&](std::uint64_t sequence, std::uint64_t stamp) {
        producer.publish(sprite_imu::encode_icm({-8192,0,0,0,0,0}, {}, {1,0,0,0},
                                                 sequence, stamp, stamp));
    };
    for (unsigned step = 1; step <= 16; ++step) publish(step, monotonic_ns());
    require(client.snapshot(s) && s.sequence == 16);
    publish(17, monotonic_ns());
    require(client.snapshot(s) && s.sequence == 17);
    publish(16, monotonic_ns());
    expect_error(client, "IMU sequence regression");
    publish(18, monotonic_ns()-31000000);
    expect_error(client, "IMU stale/future timestamp");
    producer.invalidate();
    expect_error(client, "IMU publisher stopped");
    std::cout << "Direct IMU read, latest sequence, freshness and producer stop passed\n";
}
