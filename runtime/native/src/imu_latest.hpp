#pragma once
#include "imu_packet.hpp"
#include <atomic>
#include <fcntl.h>
#include <new>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sprite_imu {
// Linux shared memory, one publisher. Each commit selects a complete slot.
// Atomic payload words also cover a reader preempted across two publications.
class LatestImu {
    using Word = std::atomic<std::uint64_t>;
    static_assert(Word::is_always_lock_free && kPacketSize % sizeof(std::uint64_t) == 0);
    static constexpr size_t words = kPacketSize / sizeof(std::uint64_t);
    static constexpr std::uint64_t stopped = UINT64_MAX;
    struct Page {
        Word generation{0};
        Word slot[2][words]{};
    };
    Page* page_;
    bool writer_;
    std::uint64_t generation_ = 0;
public:
    enum class Access { publish, read };
    LatestImu(int fd, Access mode) : writer_(mode == Access::publish) {
        struct stat info{};
        if (fcntl(fd, F_GET_SEALS) < 0 || fstat(fd, &info) != 0)
            throw std::runtime_error("IMU snapshot requires an inherited memfd");
        if (writer_) {
            if (info.st_size != 0 || ftruncate(fd, sizeof(Page)) != 0 ||
                fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) < 0)
                throw std::runtime_error("IMU snapshot initialization failed");
        } else if (info.st_size != sizeof(Page)) {
            throw std::runtime_error("IMU snapshot size mismatch");
        }
        void* mapped = mmap(nullptr, sizeof(Page), writer_ ? PROT_READ | PROT_WRITE : PROT_READ,
                            MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) throw std::runtime_error("IMU snapshot mmap failed");
        page_ = writer_ ? new (mapped) Page{} : static_cast<Page*>(mapped);
    }
    LatestImu(const LatestImu&) = delete;
    LatestImu& operator=(const LatestImu&) = delete;
    ~LatestImu() {
        if (writer_) invalidate();
        munmap(page_, sizeof(Page));
    }
    void invalidate() { page_->generation.store(stopped); }
    void publish(const Packet& packet) {
        auto next = ++generation_;
        auto& slot = page_->slot[next % 2];
        for (size_t i = 0; i < words; ++i) {
            std::uint64_t word;
            std::memcpy(&word, packet.data() + i*sizeof(word), sizeof(word));
            slot[i].store(word);
        }
        page_->generation.store(next);
    }
    bool read(Packet& packet) const {
        for (int attempt = 0; attempt < 4; ++attempt) {
            auto generation = page_->generation.load();
            if (!generation) return false;
            if (generation == stopped) throw std::runtime_error("IMU publisher stopped");
            const auto& slot = page_->slot[generation % 2];
            for (size_t i = 0; i < words; ++i) {
                auto word = slot[i].load();
                std::memcpy(packet.data() + i*sizeof(word), &word, sizeof(word));
            }
            if (page_->generation.load() == generation) return true;
        }
        throw std::runtime_error("could not read a consistent IMU snapshot");
    }
};
}
