#pragma once
#include <datareader.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace seedvr2::engine::detail {
// Own this object before Net, so referenced weights remain mapped until every
// layer and its GPU pipeline have been destroyed. Model packages are immutable
// during a run. The complete package is authenticated before this reader opens.
class MappedWeights final : public ncnn::DataReader {
public:
    explicit MappedWeights(const std::filesystem::path &path) {
#if defined(__linux__)
        fd_=open(path.c_str(),O_RDONLY|O_CLOEXEC);
        struct stat status{};
        if (fd_<0 || fstat(fd_,&status)!=0 || status.st_size<=0 || status.st_size>1024LL*1024*1024) {
            if (fd_>=0) close(fd_);
            throw std::runtime_error("Cannot map bounded model weights");
        }
        size_=static_cast<std::size_t>(status.st_size);
        void *p=mmap(nullptr,size_,PROT_READ,MAP_PRIVATE,fd_,0);
        if (p==MAP_FAILED) {close(fd_);throw std::runtime_error("Read-only weight mapping failed");}
        data_=static_cast<const unsigned char *>(p);
#else
        // Portable fallback: packages are capped at 1 GiB, so a full read is
        // bounded; mmap stays the Linux fast path.
        std::error_code ec;
        size_ = std::filesystem::file_size(path, ec);
        if (ec || size_ == 0 || size_ > 1024LL * 1024 * 1024) {
            throw std::runtime_error("Cannot read bounded model weights");
        }
        buffer_.resize(size_);
        std::ifstream in(path, std::ios::binary);
        if (!in.read(reinterpret_cast<char *>(buffer_.data()),
                     static_cast<std::streamsize>(size_))) {
            throw std::runtime_error("Read-only weight load failed");
        }
        data_ = buffer_.data();
#endif
    }
    ~MappedWeights() override {
#if defined(__linux__)
        if (data_) munmap(const_cast<unsigned char *>(data_),size_);
        if (fd_>=0) close(fd_);
#endif
    }
    MappedWeights(const MappedWeights &)=delete;
    MappedWeights &operator=(const MappedWeights &)=delete;
    std::size_t read(void *destination,std::size_t bytes) const override {
        if (bytes>size_-position_) return 0;
        std::memcpy(destination,data_+position_,bytes);position_+=bytes;return bytes;
    }
    std::size_t reference(std::size_t bytes,const void **destination) const override {
        if (bytes>size_-position_) return 0;
        *destination=data_+position_;position_+=bytes;return bytes;
    }
    bool consumed() const {return position_==size_;}
private:
    const unsigned char *data_=nullptr;
    std::size_t size_=0;
    mutable std::size_t position_=0;
    int fd_=-1;
    std::vector<unsigned char> buffer_;
};
} // namespace seedvr2::engine::detail
