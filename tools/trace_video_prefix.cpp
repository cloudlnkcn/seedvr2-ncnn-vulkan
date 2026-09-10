// Developer-only boundary intervention. This is not a complete model run.
#include "inference.hpp"
#include <iostream>
namespace d = seedvr2::engine::detail;
namespace inf = seedvr2::engine::inference;
int main(int argc, char **argv) {
    try {
        if (argc!=3) throw std::runtime_error("Usage: seedvr2-prefix-trace CASE_JSON NEW_OUTPUT");
        const std::filesystem::path file=argv[1],root=file.parent_path(),output=argv[2];
        const auto doc=d::read_document(file);
        const auto start=doc.at("start").get<std::string>();
        if (doc.at("schema_version")!="seedvr2-video-prefix-case-v1" ||
            (start!="prepared" && start!="posterior" && start!="conditioned"))
            throw std::runtime_error("Unexpected prefix intervention");
        const int t=d::integer(doc.at("padded_frames"),1,17);
        const int h=d::integer(doc.at("height"),64,128),w=d::integer(doc.at("width"),64,128);
        if ((t-1)%4 || h%16 || w%16) throw std::runtime_error("Unsupported temporal geometry");
        const int lt=(t-1)/4+1,lh=h/8,lw=w/8,gh=h/16,gw=w/16,tokens=lt*gh*gw;
        auto read=[&](const char *key,const std::vector<int> &shape) {
            const auto &spec=doc.at(key);
            if (spec.at("shape")!=shape || spec.at("dtype")!="f32le") throw std::runtime_error("Invalid prefix tensor");
            auto value=d::allocate_tensor(shape);d::read_tensor(d::artifact(root,spec),value);return value;
        };
        auto noise=read("noise",{16,lt,lh,lw});
        auto posterior_noise=read("posterior_noise",{16,lt,lh,lw});
        auto input=read("input",start=="prepared"?std::vector<int>{3,t,h,w}:
                       std::vector<int>{start=="posterior"?32:16,lt,lh,lw});
        if (!std::filesystem::create_directories(output)) throw std::runtime_error("Output already exists");
        d::init_gpu();const int gpu=0;
        seedvr2::engine::ImageRequest request;
        request.vulkan=true;request.gpu_index=gpu;request.threads=4;
        ncnn::PipelineCache pipelines(ncnn::get_gpu_device(gpu));
        d::Json stages=d::Json::array(),captures=d::Json::object();
        auto capture=[&](const char *name,const ncnn::Mat &value) {
            auto row=d::write_tensor(output,(std::string(name)+".f32").c_str(),value);
            row["shape"]=d::tensor_shape(value);captures[name]=row;
        };
        auto execute=[&](const char *name,const ncnn::Mat &value) {
            const auto &graph=doc.at("graphs").at(name);
            const inf::GraphFiles files{d::artifact(root,graph.at("param")),d::artifact(root,graph.at("weights"),1024ULL*1024*1024)};
            return inf::execute_graph(files,name,{value},request,gpu,stages,[]{},&pipelines)[0];
        };
        ncnn::Mat conditioned;
        if (start=="conditioned") conditioned=input;
        else {
            const auto posterior=start=="prepared"?execute("encoder",input):input;
            inf::expect(posterior,{32,lt,lh,lw},"posterior");capture("posterior",posterior);
            conditioned.create(lw,lh,lt,16,size_t(4),1);
            if (conditioned.empty()) throw std::runtime_error("Cannot allocate condition");
            for (int c=0;c<16;++c) {
                auto p=posterior_noise.channel(c),z=conditioned.channel(c),mean=posterior.channel(c),logvar=posterior.channel(c+16);
                for (int j=0;j<lt*lh*lw;++j)
                    z[j]=(mean[j]+std::exp(.5f*std::clamp(logvar[j],-30.f,20.f))*p[j])*.9152f;
            }
        }
        capture("conditioned",conditioned);
        ncnn::Mat patches(132,tokens,size_t(4),1);
        if (patches.empty()) throw std::runtime_error("Cannot allocate patches");
        for (int t0=0;t0<lt;++t0) for (int y=0;y<gh;++y) for (int x=0;x<gw;++x) {
            auto *p=patches.row((t0*gh+y)*gw+x);
            for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx) {
                const int offset=(dy*2+dx)*33,index=(t0*lh+2*y+dy)*lw+2*x+dx;
                for (int c=0;c<16;++c) {p[offset+c]=noise.channel(c)[index];p[offset+c+16]=conditioned.channel(c)[index];}
                p[offset+32]=1.f;
            }
        }
        capture("patches",patches);const auto projected=execute("patch-in",patches);
        inf::expect(projected,{tokens,2560},"patch-in");capture("patch-in",projected);
        d::Json report{{"schema_version","seedvr2-video-prefix-trace-v1"},{"status","EXECUTED"},
            {"diagnostic_only",true},{"model_verified",false},{"case_sha256",d::hash(file)},
            {"case",doc},{"ncnn_commit",SEEDVR2_NCNN_COMMIT},{"device",d::device_info(gpu)},
            {"outputs",captures},{"stages",stages}};
#if defined(__linux__)
        report["executable_sha256"]=d::hash("/proc/self/exe");
#endif
        std::ofstream stream(output/"report.json");stream<<report.dump(2)<<'\n';
        if (!stream) throw std::runtime_error("Cannot write prefix report");
        std::cout<<"Recorded native prefix from "<<start<<'\n';
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 2;}
}
