// Standalone investigation tool; it does not enable low precision in the SDK.
#include "graph.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace d = seedvr2::engine::detail;
using d::Json;
using Path = std::filesystem::path;

static Json options(const ncnn::Option &o) {
    return {{"fp16_storage",o.use_fp16_storage},{"fp16_packed",o.use_fp16_packed},
        {"fp16_arithmetic",o.use_fp16_arithmetic},{"fp16_uniform",o.use_fp16_uniform},
        {"bf16_storage",o.use_bf16_storage},{"bf16_packed",o.use_bf16_packed},
        {"cooperative_matrix",o.use_cooperative_matrix},{"subgroups",o.use_subgroup_ops}};
}
static ncnn::Option precision(const std::string &name) {
    ncnn::Option o;
    o.use_vulkan_compute=true; o.num_threads=1;
    o.use_fp16_storage=o.use_fp16_packed=o.use_fp16_arithmetic=o.use_fp16_uniform=false;
    o.use_bf16_storage=o.use_bf16_packed=false;
    o.use_int8_inference=o.use_int8_storage=o.use_int8_packed=o.use_int8_arithmetic=false;
    o.use_int8_uniform=o.use_int16_packed=o.use_int16_storage=false;
    o.use_cooperative_matrix=o.use_winograd_convolution=false;
    if(name=="fp16-storage") o.use_fp16_storage=true;
    else if(name=="fp16-arithmetic") o.use_fp16_storage=o.use_fp16_arithmetic=o.use_fp16_uniform=true;
    else if(name=="fp16-packed-arithmetic") o.use_fp16_packed=o.use_fp16_arithmetic=o.use_fp16_uniform=true;
    else if(name=="bf16-storage") o.use_bf16_storage=true;
    else if(name=="bf16-packed") o.use_bf16_packed=true;
    else if(name!="fp32") throw std::runtime_error("Unknown precision");
    return o;
}
static std::string unavailable(const ncnn::GpuInfo &g, const ncnn::Option &o) {
    if(o.use_fp16_storage && !g.support_fp16_storage()) return "fp16 storage unavailable";
    if(o.use_fp16_packed && !g.support_fp16_packed()) return "fp16 packed unavailable";
    if(o.use_fp16_arithmetic && !g.support_fp16_arithmetic()) return "fp16 arithmetic unavailable";
    if(o.use_fp16_uniform && !g.support_fp16_uniform()) return "fp16 uniform unavailable";
    if(o.use_bf16_storage && !g.support_bf16_storage()) return "bf16 storage unavailable";
    if(o.use_bf16_packed && !g.support_bf16_packed()) return "bf16 packed unavailable";
    return {};
}
static Json layout(const ncnn::VkMat &m) {
    return {{"dims",m.dims},{"w",m.w},{"h",m.h},{"d",m.d},{"c",m.c},
        {"elempack",m.elempack},{"elemsize",m.elemsize},
        {"scalar_storage_bits",8*m.elemsize/static_cast<size_t>(m.elempack)}};
}
static std::vector<float> flatten(const ncnn::Mat &m) {
    if(m.empty() || m.n!=1 || m.elempack!=1 || m.elemsize!=4) throw std::runtime_error("Output is not unpacked FP32");
    std::vector<float> out;
    const size_t n=static_cast<size_t>(m.w)*m.h*m.d;
    for(int c=0;c<(m.dims>=3?m.c:1);++c) {
        const float *p=static_cast<const float *>(m.data)+static_cast<size_t>(c)*m.cstep;
        out.insert(out.end(),p,p+n);
    }
    return out;
}
static Json metrics(const std::vector<float> &got,const std::vector<float> &ref) {
    if(got.size()!=ref.size() || ref.empty()) throw std::runtime_error("Output element contract differs");
    double maximum=0, square=0, normalized=0;
    size_t violations=0, nonfinite=0;
    for(size_t i=0;i<ref.size();++i) {
        if(!std::isfinite(got[i])) {++nonfinite;++violations;continue;}
        const double delta=std::abs(double(got[i])-ref[i]);
        const double limit=.001+.001*std::abs(double(ref[i]));
        maximum=std::max(maximum,delta);square+=delta*delta;
        normalized=std::max(normalized,delta/limit);violations+=delta>limit;
    }
    return {{"elements",ref.size()},{"nonfinite",nonfinite},{"violations",violations},
        {"max_abs",nonfinite?Json(nullptr):Json(maximum)},
        {"rmse",nonfinite?Json(nullptr):Json(std::sqrt(square/ref.size()))},
        {"max_error_over_limit",nonfinite?Json(nullptr):Json(normalized)},
        {"passed",violations==0}};
}
static Json raw(const Path &root,const std::string &name,const std::vector<float> &v) {
    const Path p=root/name;
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char *>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(float)));f.close();
    if(!f) throw std::runtime_error("Cannot retain tensor");
    return {{"path",name},{"elements",v.size()},{"dtype","f32le"},{"sha256",d::hash(p)}};
}
static ncnn::Mat matrix(int w,int h) { return ncnn::Mat(w,h,size_t(4),1); }

int main(int argc,char **argv) {
    Json report={{"schema_version","ncnn-precision-probe-v1"},{"ncnn_commit",PROBE_NCNN_COMMIT},
        {"ncnn_library_sha256",PROBE_NCNN_LIBRARY_SHA256},{"model_verified",false},
        {"tolerance",{{"atol",.001},{"rtol",.001},{"calibration","FP32_DIAGNOSTIC_NOT_LOW_PRECISION_CERTIFICATION"}}}};
    try {
        if(argc!=5 && argc!=6) throw std::runtime_error("Usage: probe CASE PRECISION GPU NEW_OUTPUT_DIR [GRAPH_CASE_JSON]");
        const std::string name=argv[1], mode=argv[2];
        size_t end=0; const int gpu=std::stoi(argv[3],&end);
        if(end!=std::string(argv[3]).size() || gpu<0) throw std::runtime_error("Invalid GPU index");
        const Path output=argv[4];
        if(!std::filesystem::create_directories(output)) throw std::runtime_error("Output directory already exists");
        report["case"]=name;report["precision"]=mode;
        auto opt=precision(mode);
        if(name=="reduction-local") opt.use_subgroup_ops=false;
        report["requested_options"]=options(opt);
        d::init_gpu();
        if(gpu>=ncnn::get_gpu_count()) throw std::runtime_error("GPU index unavailable");
        const auto *device=ncnn::get_gpu_device(gpu);
        report["device"]=d::device_info(gpu);
        const auto reason=unavailable(device->info,opt);
        if(!reason.empty()) {report["status"]="SKIPPED";report["reason"]=reason;std::cout<<report.dump(2)<<'\n';return 77;}
        std::vector<ncnn::Mat> inputs;
        std::vector<float> reference;
        std::string param;
        Json graph;
        Path graph_root;
        if(name=="patch-in") {
            if(argc!=6) throw std::runtime_error("Real component case required");
            const Path case_path=argv[5];graph_root=case_path.parent_path();graph=d::read_document(case_path);
            if(graph.at("schema_version")!="seedvr2-patch-in-precision-case-v1" || graph.at("input").at("shape")!=Json::array({320,132}) || graph.at("reference").at("shape")!=Json::array({320,2560}))
                throw std::runtime_error("Unexpected real component contract");
            inputs.push_back(matrix(132,320));
            d::read_tensor(d::artifact(graph_root,graph.at("input")),inputs[0]);
            auto target=matrix(2560,320);d::read_tensor(d::artifact(graph_root,graph.at("reference")),target);
            reference=flatten(target);report["case_sha256"]=d::hash(case_path);report["reference_kind"]="REVIEWED_OFFICIAL_FP32_B_SAME_INPUT";
        } else if(name=="sdpa") {
            constexpr int heads=20,tokens=65,width=128;
            for(int k=0;k<3;++k) {
                ncnn::Mat m(width,tokens,heads,size_t(4),1);
                for(int h=0;h<heads;++h) for(int t=0;t<tokens;++t) for(int x=0;x<width;++x)
                    m.channel(h).row(t)[x]=float(((h*13+t*7+x*3+k*19)%127)-63)/64.f;
                inputs.push_back(m);
            }
            reference.resize(heads*tokens*width);
            for(int h=0;h<heads;++h) for(int t=0;t<tokens;++t) {
                std::vector<double> scores(tokens);
                for(int s=0;s<tokens;++s) {
                    double sum=0;
                    for(int x=0;x<width;++x) sum+=double(inputs[0].channel(h).row(t)[x])*inputs[1].channel(h).row(s)[x];
                    scores[s]=sum/std::sqrt(double(width));
                }
                const double top=*std::max_element(scores.begin(),scores.end());double denom=0;
                for(auto &x:scores) {x=std::exp(x-top);denom+=x;}
                for(int x=0;x<width;++x) {
                    double sum=0;for(int s=0;s<tokens;++s) sum+=scores[s]*inputs[2].channel(h).row(s)[x];
                    reference[(h*tokens+t)*width+x]=float(sum/denom);
                }
            }
            param="7767517\n4 4\nInput in0 0 1 in0\nInput in1 0 1 in1\nInput in2 0 1 in2\nSDPA attention 3 1 in0 in1 in2 out0\n";
            report["reference_kind"]="ANALYTIC_FP64_SYNTHETIC_20_HEADS_128_WIDTH_NOT_OFFICIAL_AWA";
        } else {
            const bool reduction=name=="reduction" || name=="reduction-local" || name=="reduction-cancellation";
            if(!reduction && name!="erf" && name!="celu") throw std::runtime_error("Unknown case");
            const int n=reduction?65536:1024;ncnn::Mat m(n);
            double sum=0;
            for(int i=0;i<n;++i) {
                // Exactly BF16-representable inputs; positive/negative workgroup
                // partial sums cancel, leaving 128 * 1/256 = 0.5. Saving FP32
                // subgroup partial sums in BF16 loses this small remainder.
                if(name=="reduction-cancellation") m[i]=i%256<128?.5f+(i<128?1.f/256.f:0.f):-.5f;
                else m[i]=reduction?float((i%31)-15)/32.f+1.f/256.f:float(i-512)/128.f;
                sum+=m[i];
                if(!reduction) reference.push_back(float(name=="erf"?std::erf(double(m[i])):std::max(double(m[i]),0.)+std::min(.5*std::expm1(double(m[i])/.5),0.)));
            }
            if(reduction) reference.push_back(float(sum));
            inputs.push_back(m);
            const std::string layer=reduction?"Reduction op 1 1 in0 out0 0=0 1=1 2=1.0":name=="erf"?"Erf op 1 1 in0 out0":"CELU op 1 1 in0 out0 0=0.5";
            param="7767517\n2 2\nInput in0 0 1 in0\n"+layer+"\n";
            report["reference_kind"]="ANALYTIC_FP64_SYNTHETIC_NOT_MODEL";
        }
        report["inputs"]=Json::array();
        for(size_t i=0;i<inputs.size();++i) report["inputs"].push_back(raw(output,"input-"+std::to_string(i)+".f32",flatten(inputs[i])));
        report["reference"]=raw(output,"reference.f32",reference);
        ncnn::Net net;net.opt=opt;net.set_vulkan_device(gpu);
        const auto started=d::Clock::now();
        const int loaded=name=="patch-in"?net.load_param(d::artifact(graph_root,graph.at("param")).string().c_str()):net.load_param_mem(param.c_str());
        if(loaded!=0) throw std::runtime_error("Graph parameter load failed");
        if(options(net.opt)!=options(opt)) throw std::runtime_error("ncnn changed requested precision; refusing silent downgrade");
        for(const auto *layer:net.layers()) if(layer->type!="Input" && !layer->support_vulkan) throw std::runtime_error("Required Vulkan layer absent");
        alignas(4) static const unsigned char empty[4]={};
        const int weights=name=="patch-in"?net.load_model(d::artifact(graph_root,graph.at("weights")).string().c_str()):net.load_model(empty);
        if(weights<0) throw std::runtime_error("Graph weights/pipeline load failed");
        report["effective_options"]=options(net.opt);
        ncnn::Mat result;
        {
            d::VulkanAllocators allocators(device);
            opt=net.opt;opt.blob_vkallocator=opt.workspace_vkallocator=allocators.blob;opt.staging_vkallocator=allocators.staging;
            ncnn::VkCompute command(device);std::vector<ncnn::VkMat> uploaded(inputs.size());
            report["uploaded_layouts"]=Json::array();
            for(size_t i=0;i<inputs.size();++i) {
                command.record_upload(inputs[i],uploaded[i],opt);
                if(uploaded[i].empty()) throw std::runtime_error("Upload failed");
                report["uploaded_layouts"].push_back(layout(uploaded[i]));
                const auto bits=8*uploaded[i].elemsize/static_cast<size_t>(uploaded[i].elempack);
                if(bits!=(mode=="fp32"?32u:16u)) throw std::runtime_error("Requested input storage precision did not execute");
            }
            Json trace=Json::array();auto results=d::graph_forward(net,uploaded,&command,opt,trace);
            if(results.size()!=1) throw std::runtime_error("Expected exactly one output");
            auto value=results[0];report["gpu_output_layout"]=layout(value);
            if(value.elempack!=1) {ncnn::VkMat one;device->convert_packing(value,one,1,command,opt);value=one;}
            auto download=opt;download.use_packing_layout=false;
            command.record_download(value,result,download);
            if(command.submit_and_wait()!=0) throw std::runtime_error("GPU submission failed");
            report["layers"]=trace;report["dispatch"]="EXPLICIT_VULKAN_NO_CPU_FALLBACK";
        }
        report["load_upload_compute_download_ms"]=std::chrono::duration<double,std::milli>(d::Clock::now()-started).count();
        const bool shape_ok=name=="patch-in"?(result.dims==2 && result.w==2560 && result.h==320):
            name=="sdpa"?(result.dims==3 && result.w==128 && result.h==65 && result.c==20):
            (result.dims==1 && result.w==(name.starts_with("reduction")?1:1024));
        if(!shape_ok) throw std::runtime_error("Output shape contract differs");
        const auto values=flatten(result);report["output"]=raw(output,"output.f32",values);
        report["metrics"]=metrics(values,reference);report["status"]="EXECUTED";
        std::cout<<report.dump(2)<<'\n';return report["metrics"]["passed"]==true?0:1;
    } catch(const std::exception &e) {
        report["status"]="ERROR";report["reason"]=e.what();std::cout<<report.dump(2)<<'\n';return 2;
    }
}
