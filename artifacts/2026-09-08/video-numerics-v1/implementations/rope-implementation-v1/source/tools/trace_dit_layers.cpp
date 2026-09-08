// Developer-only trace using the same graph dispatcher and AWA implementation.
#include "graph.hpp"
#include "awa.hpp"
#include "constant.hpp"
#include "engine_build.hpp"
#include <iostream>

namespace d = seedvr2::engine::detail;
namespace e = seedvr2::engine;
using d::Json;

int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: seedvr2-dit-trace CASE_JSON cpu|vulkan NEW_OUTPUT");
        const std::filesystem::path file=argv[1], root=file.parent_path(), output=argv[3];
        const std::string backend=argv[2];
        if (backend!="cpu" && backend!="vulkan") throw std::runtime_error("Unknown backend");
        const auto doc=d::read_document(file);
        if (doc.at("schema_version")!="dit-block-case-v1" || doc.at("precision")!="fp32" ||
            doc.at("inputs").size()!=3) throw std::runtime_error("Expected an FP32 DiT case");
        const int index=d::integer(doc.at("block_index"),0,31);
        if (!std::filesystem::create_directories(output)) throw std::runtime_error("Output already exists");
        ncnn::Net net;
        auto &opt=net.opt;
        opt.use_vulkan_compute=backend=="vulkan";
        opt.use_packing_layout=false;
        opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;
        opt.use_bf16_storage=opt.use_bf16_packed=false;
        opt.use_winograd_convolution=opt.use_cooperative_matrix=false;
        opt.num_threads=4;
        if (opt.use_vulkan_compute) {d::init_gpu();net.set_vulkan_device(0);}
        e::ExecutionTrace awa;
        const auto param=d::artifact(root,doc.at("model_param"));
        const auto weights=d::artifact(root,doc.at("model_bin"),1024ULL*1024*1024);
        if (e::register_awa(net,awa) || e::register_constant(net) || net.load_param(param.c_str()) ||
            net.layers().size()>512 || net.input_indexes().size()!=3 || net.output_indexes().size()!=2 ||
            awa.heads!=20 || awa.shifted!=index%2 || net.load_model(weights.c_str()))
            throw std::runtime_error("Cannot load expected DiT graph");
        std::vector<ncnn::Mat> inputs;
        const std::vector<std::vector<int>> shapes{{5,8,8,2560},{58,2560},{2560,6}};
        for (std::size_t i=0;i<3;++i) {
            const auto &row=doc.at("inputs").at(i);
            if (row.at("shape")!=shapes[i] || row.at("dtype")!="f32le") throw std::runtime_error("Unexpected input contract");
            auto tensor=d::allocate_tensor(shapes[i]);d::read_tensor(d::artifact(root,row),tensor);
            inputs.push_back(tensor);
        }
        Json report{{"schema_version","seedvr2-dit-layer-trace-v1"},{"case_sha256",d::hash(file)},
            {"ncnn_commit",SEEDVR2_NCNN_COMMIT},{"backend",backend},{"model_verified",false},
            {"executable_sha256",d::hash(std::filesystem::canonical("/proc/self/exe"))},
            {"layers",Json::array()}};
        Json trace=Json::array();
        int serial=0;
        auto save=[&](const ncnn::Layer &layer,const std::vector<ncnn::Mat> &tops) {
            Json row{{"name",layer.name},{"type",layer.type},{"outputs",Json::array()}};
            for (std::size_t i=0;i<tops.size();++i) {
                const auto name="tensor-"+std::to_string(serial++)+".f32";
                auto item=d::write_tensor(output,name.c_str(),tops[i]);
                item["shape"]=d::tensor_shape(tops[i]);row["outputs"].push_back(item);
            }
            report["layers"].push_back(row);
        };
        const std::set<std::string> selected{"RMSNorm","InnerProduct","SeedVR2AWA","Swish","BinaryOp"};
        awa.observe_cpu=[&](const char *name,int window,const std::vector<ncnn::Mat> &tops) {
            ncnn::Layer layer;layer.name=std::string(name)+"."+std::to_string(window);layer.type="AWA_INTERNAL";
            save(layer,tops);
        };
        if (opt.use_vulkan_compute) {
            const auto *device=net.vulkan_device();d::VulkanAllocators allocators(device);
            auto options=opt;options.blob_vkallocator=options.workspace_vkallocator=allocators.blob;
            options.staging_vkallocator=allocators.staging;
            ncnn::VkCompute command(device);std::vector<ncnn::VkMat> uploaded(3);
            for (std::size_t i=0;i<3;++i) command.record_upload(inputs[i],uploaded[i],options);
            auto observe=[&](const ncnn::Layer &layer,const std::vector<ncnn::VkMat> &tops) {
                if (!selected.contains(layer.type) && layer.type!="AWA_INTERNAL") return;
                std::vector<ncnn::Mat> downloaded(tops.size());
                for (std::size_t i=0;i<tops.size();++i) {
                    auto value=tops[i];
                    if (value.elempack!=1) {ncnn::VkMat one;device->convert_packing(value,one,1,command,options);value=one;}
                    command.record_download(value,downloaded[i],options);
                }
                if (command.submit_and_wait()!=0 || command.reset()!=0) throw std::runtime_error("Trace download failed");
                save(layer,downloaded);
            };
            awa.observe_vulkan=[&](const char *name,int window,const std::vector<ncnn::VkMat> &tops,
                                  ncnn::VkCompute &,const ncnn::Option &) {
                ncnn::Layer layer;layer.name=std::string(name)+"."+std::to_string(window);layer.type="AWA_INTERNAL";
                observe(layer,tops);
            };
            auto out=d::graph_forward<ncnn::VkMat>(net,uploaded,&command,options,trace,{},0,observe);
            if (command.submit_and_wait()!=0) throw std::runtime_error("Trace submission failed");
        } else {
            auto observe=[&](const ncnn::Layer &layer,const std::vector<ncnn::Mat> &tops) {
                if (selected.contains(layer.type)) save(layer,tops);
            };
            auto out=d::graph_forward<ncnn::Mat>(net,inputs,nullptr,opt,trace,{},0,observe);
        }
        report["dispatch"]=trace;report["status"]="EXECUTED";
        std::ofstream stream(output/"report.json");stream<<report.dump(2)<<'\n';
        std::cout<<"Recorded "<<report["layers"].size()<<" layer boundaries\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr<<error.what()<<'\n';return 2;
    }
}
