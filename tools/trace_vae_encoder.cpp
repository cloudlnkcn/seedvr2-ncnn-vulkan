// Bounded CPU/Vulkan diagnostic for a retained temporal encoder input.
#include "graph.hpp"
#include "video_layers.hpp"
#include "silu.hpp"
#include "engine_build.hpp"
#include <iostream>
#include <sstream>
namespace d = seedvr2::engine::detail;
int main(int argc, char **argv) {
    try {
        if (argc != 4 && argc != 5) throw std::runtime_error("Usage: seedvr2-vae-trace CASE_JSON sgemm|direct|vulkan|vulkan-direct NEW_OUTPUT [EXTRA_LAYER_NAMES_COMMA_SEPARATED]");
        const std::filesystem::path file=argv[1], root=file.parent_path(), output=argv[3];
        const std::string mode=argv[2];
        if (mode!="sgemm" && mode!="direct" && mode!="vulkan" && mode!="vulkan-direct") throw std::runtime_error("Unknown convolution mode");
        const auto doc=d::read_document(file);
        const auto shape=doc.at("input").at("shape").get<std::vector<int>>();
        if (doc.at("schema_version")!="ncnn-graph-case-v1" || doc.at("component")!="vae-video-encoder" ||
            doc.at("precision")!="fp32" || doc.at("reference_profile")!="FP32-B" ||
            shape.size()!=4 || shape[0]!=3 || shape[1]<1 || shape[1]>17 || (shape[1]-1)%4 ||
            shape[2]<8 || shape[2]>128 || shape[2]%8 || shape[3]<8 || shape[3]>128 || shape[3]%8 ||
            doc.at("input").at("dtype")!="f32le") throw std::runtime_error("Unexpected encoder contract");
        const auto param=d::artifact(root,doc.at("model_param"));
        const auto weights=d::artifact(root,doc.at("model_bin"),1024ULL*1024*1024);
        auto input=d::allocate_tensor(shape);d::read_tensor(d::artifact(root,doc.at("input")),input);
        if (!std::filesystem::create_directories(output)) throw std::runtime_error("Output already exists");
        ncnn::Net net;auto &opt=net.opt;
        opt.use_vulkan_compute=mode.starts_with("vulkan");opt.use_packing_layout=false;
        opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;
        opt.use_bf16_storage=opt.use_bf16_packed=false;
        opt.use_winograd_convolution=opt.use_cooperative_matrix=false;
        opt.use_sgemm_convolution=mode!="direct" && mode!="vulkan-direct";opt.num_threads=4;
        if (opt.use_vulkan_compute) { d::init_gpu();net.set_vulkan_device(0); }
        if (seedvr2::engine::register_video_layers(net) || seedvr2::engine::register_dit_silu(net) || net.load_param(param.c_str()) ||
            net.layers().size()!=99 || net.input_indexes().size()!=1 || net.output_indexes().size()!=1 ||
            net.load_model(weights.c_str())) throw std::runtime_error("Cannot load reviewed encoder");
        d::Json report{{"schema_version","seedvr2-vae-encoder-trace-v1"},{"case_sha256",d::hash(file)},
            {"ncnn_commit",SEEDVR2_NCNN_COMMIT},{"backend",opt.use_vulkan_compute?"vulkan":"cpu"},
            {"convolution",mode},{"input_shape",shape},{"threads",4},
            {"model_verified",false},{"layers",d::Json::array()}};
#if defined(__linux__)
        report["executable_sha256"]=d::hash("/proc/self/exe");
#endif
        std::set<std::string> selected{"conv_in","norm1","add_10","pnnx_unique_45","pnnx_unique_46"};
        if (argc==5) {
            std::istringstream names(argv[4]);std::string name;
            while (std::getline(names,name,',')) {
                const auto &layers=net.layers();
                if (name.empty() || std::none_of(layers.begin(),layers.end(),[&](const ncnn::Layer *layer) {return layer->name==name;}))
                    throw std::runtime_error("Unknown extra layer: "+name);
                selected.insert(name);
            }
            if (selected.size()>24) throw std::runtime_error("Too many trace layers");
        }
        report["selected_layers"]=selected;
        auto observe=[&](const ncnn::Layer &layer,const std::vector<ncnn::Mat> &tops) {
            if (!selected.contains(layer.name)) return;
            if (tops.size()!=1) throw std::runtime_error("Unexpected selected output arity");
            auto row=d::write_tensor(output,(layer.name+".f32").c_str(),tops[0]);
            row["shape"]=d::tensor_shape(tops[0]);row["name"]=layer.name;row["type"]=layer.type;
            report["layers"].push_back(row);
        };
        const auto started=d::Clock::now();d::Json dispatch=d::Json::array();
        std::vector<ncnn::Mat> out;
        if (opt.use_vulkan_compute) {
            const auto *device=net.vulkan_device();d::VulkanAllocators allocators(device);
            auto options=opt;options.blob_vkallocator=options.workspace_vkallocator=allocators.blob;
            options.staging_vkallocator=allocators.staging;
            ncnn::VkCompute command(device);ncnn::VkMat uploaded;
            command.record_upload(input,uploaded,options);
            auto observe_gpu=[&](const ncnn::Layer &layer,const std::vector<ncnn::VkMat> &tops) {
                if (!selected.contains(layer.name)) return;
                std::vector<ncnn::Mat> downloaded(tops.size());
                for (std::size_t i=0;i<tops.size();++i) {
                    auto value=tops[i];
                    if (value.elempack!=1) { ncnn::VkMat one;device->convert_packing(value,one,1,command,options);value=one; }
                    command.record_download(value,downloaded[i],options);
                }
                if (command.submit_and_wait()!=0 || command.reset()!=0) throw std::runtime_error("Trace download failed");
                observe(layer,downloaded);
            };
            auto gpu_out=d::graph_forward<ncnn::VkMat>(net,{uploaded},&command,options,dispatch,{},4,observe_gpu);
            out.resize(gpu_out.size());
            for (std::size_t i=0;i<gpu_out.size();++i) command.record_download(gpu_out[i],out[i],options);
            if (command.submit_and_wait()!=0) throw std::runtime_error("Trace submission failed");
            report["device"]=d::device_info(0);
        } else {
            out=d::graph_forward<ncnn::Mat>(net,{input},nullptr,opt,dispatch,{},0,observe);
        }
        report["elapsed_seconds"]=std::chrono::duration<double>(d::Clock::now()-started).count();
        if (out.size()!=1 || report["layers"].size()!=selected.size()) throw std::runtime_error("Incomplete trace");
        report["status"]="EXECUTED";report["dispatch"]=dispatch;
        std::ofstream stream(output/"report.json");stream<<report.dump(2)<<'\n';
        if (!stream) throw std::runtime_error("Cannot write trace report");
        std::cout<<"Recorded "<<report["layers"].size()<<" encoder boundaries\n";
    } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 2;}
}
