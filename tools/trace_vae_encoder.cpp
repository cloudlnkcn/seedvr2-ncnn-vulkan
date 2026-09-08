// Bounded CPU diagnostic for the reviewed 17-frame encoder input.
#include "graph.hpp"
#include "video_layers.hpp"
#include "engine_build.hpp"
#include <iostream>
namespace d = seedvr2::engine::detail;
int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: seedvr2-vae-trace CASE_JSON sgemm|direct NEW_OUTPUT");
        const std::filesystem::path file=argv[1], root=file.parent_path(), output=argv[3];
        const std::string mode=argv[2];
        if (mode!="sgemm" && mode!="direct") throw std::runtime_error("Unknown convolution mode");
        const auto doc=d::read_document(file);
        const std::vector<int> shape{3,17,128,128};
        if (doc.at("schema_version")!="ncnn-graph-case-v1" || doc.at("component")!="vae-video-encoder" ||
            doc.at("precision")!="fp32" || doc.at("input").at("shape")!=shape ||
            doc.at("input").at("dtype")!="f32le") throw std::runtime_error("Unexpected encoder contract");
        const auto param=d::artifact(root,doc.at("model_param"));
        const auto weights=d::artifact(root,doc.at("model_bin"),1024ULL*1024*1024);
        auto input=d::allocate_tensor(shape);d::read_tensor(d::artifact(root,doc.at("input")),input);
        if (!std::filesystem::create_directories(output)) throw std::runtime_error("Output already exists");
        ncnn::Net net;auto &opt=net.opt;
        opt.use_vulkan_compute=opt.use_packing_layout=false;
        opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;
        opt.use_bf16_storage=opt.use_bf16_packed=false;
        opt.use_winograd_convolution=opt.use_cooperative_matrix=false;
        opt.use_sgemm_convolution=mode=="sgemm";opt.num_threads=4;
        if (seedvr2::engine::register_video_layers(net) || net.load_param(param.c_str()) ||
            net.layers().size()!=99 || net.input_indexes().size()!=1 || net.output_indexes().size()!=1 ||
            net.load_model(weights.c_str())) throw std::runtime_error("Cannot load reviewed encoder");
        d::Json report{{"schema_version","seedvr2-vae-encoder-trace-v1"},{"case_sha256",d::hash(file)},
            {"ncnn_commit",SEEDVR2_NCNN_COMMIT},{"backend","cpu"},{"convolution",mode},{"threads",4},
            {"model_verified",false},{"layers",d::Json::array()}};
#if defined(__linux__)
        report["executable_sha256"]=d::hash("/proc/self/exe");
#endif
        const std::set<std::string> selected{"conv_in","norm1","add_10","pnnx_unique_45","pnnx_unique_46"};
        auto observe=[&](const ncnn::Layer &layer,const std::vector<ncnn::Mat> &tops) {
            if (!selected.contains(layer.name)) return;
            if (tops.size()!=1) throw std::runtime_error("Unexpected selected output arity");
            auto row=d::write_tensor(output,(layer.name+".f32").c_str(),tops[0]);
            row["shape"]=d::tensor_shape(tops[0]);row["name"]=layer.name;row["type"]=layer.type;
            report["layers"].push_back(row);
        };
        const auto started=d::Clock::now();d::Json dispatch=d::Json::array();
        auto out=d::graph_forward<ncnn::Mat>(net,{input},nullptr,opt,dispatch,{},0,observe);
        report["elapsed_seconds"]=std::chrono::duration<double>(d::Clock::now()-started).count();
        if (out.size()!=1 || report["layers"].size()!=selected.size()) throw std::runtime_error("Incomplete trace");
        report["status"]="EXECUTED";report["dispatch"]=dispatch;
        std::ofstream stream(output/"report.json");stream<<report.dump(2)<<'\n';
        if (!stream) throw std::runtime_error("Cannot write trace report");
        std::cout<<"Recorded "<<report["layers"].size()<<" encoder boundaries\n";
    } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 2;}
}
