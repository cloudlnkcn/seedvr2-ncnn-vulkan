#include "memory.hpp"
#include <layer/vulkan/gemm_vulkan.h>
#include <array>
#include <fstream>
#include <iostream>

namespace d=seedvr2::engine::detail;
namespace {
void require(bool result,const char *message){if(!result)throw std::runtime_error(message);}
struct Temporary {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("seedvr2-memory-"+std::to_string(d::Clock::now().time_since_epoch().count()));
    Temporary(){require(std::filesystem::create_directory(path),"Cannot create test directory");}
    ~Temporary(){std::error_code ignored;std::filesystem::remove_all(path,ignored);}
};
}
int main() {
    try {
        try {d::init_gpu();}catch(const std::exception &e){std::cout<<"SKIP: "<<e.what()<<'\n';return 77;}
        if(!ncnn::get_gpu_count()){std::cout<<"SKIP: no Vulkan device\n";return 77;}
        Temporary temporary;
        const auto weights=temporary.path/std::filesystem::path(u8"权重 空格.bin");
        {
            std::ofstream file(weights,std::ios::binary);
            const std::uint32_t tag=0;
            const std::array<float,4> data{2.f,0.f,0.f,3.f};
            file.write(reinterpret_cast<const char *>(&tag),sizeof(tag));
            file.write(reinterpret_cast<const char *>(data.data()),sizeof(data));
            require(bool(file),"Cannot write small weights");
        }
        const auto *device=ncnn::get_gpu_device(0);
        d::VulkanAllocators allocators(device);
        ncnn::Option option;
        option.use_vulkan_compute=true;
        option.use_packing_layout=false;
        option.use_fp16_storage=option.use_fp16_packed=option.use_fp16_arithmetic=false;
        option.use_bf16_storage=option.use_bf16_packed=false;
        option.use_cooperative_matrix=false;
        option.blob_vkallocator=option.workspace_vkallocator=allocators.blob;
        option.staging_vkallocator=allocators.staging;
        ncnn::Mat input(2,1,size_t(4));input[0]=1.25f;input[1]=-2.f;
        ncnn::VkMat current;
        {ncnn::VkCompute cmd(device);cmd.record_upload(input,current,option);require(!cmd.submit_and_wait(),"Input upload failed");}
        d::Json results=d::Json::array();
        int queries=0;
        auto controlled=[&](const ncnn::VulkanDevice *) -> std::optional<seedvr2::memory::DeviceBudget> {
            ++queries;
            if(queries==4)return std::nullopt;
            return seedvr2::memory::DeviceBudget{200,queries==2?170u:0u,0};
        };
        for(int step=0;step<4;++step) {
            ncnn::Net net;net.set_vulkan_device(0);net.opt=option;
            const char *graph="7767517\n2 2\nInput in0 0 1 in0\n"
                "Gemm projection 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=1 8=2 9=2 10=-1\n";
            require(!net.load_param_mem(graph),"Cannot load GEMM graph");
            auto placement=d::configure_memory(net,weights,{seedvr2::WeightPlacement::automatic,20},controlled);
            const bool host=step==1||step==3;
            require(net.opt.use_weights_in_host_memory==host,"Incorrect GPU/RAM transition");
            require(!net.load_model(weights.string().c_str()),"Cannot load GEMM weights");
            // Net resolves device capabilities while loading the graph (e.g.
            // disables shader local memory on Mesa). Dispatch with that same
            // option set, as the application does, not the original defaults.
            const auto runtime_option=net.opt;
            const auto *gemm=dynamic_cast<const ncnn::Gemm_vulkan *>(net.layers().back());
            require(gemm&&!gemm->B_data_gpu.empty(),"GEMM did not upload real weights");
            placement["observed_weight_device_local"]=device->is_device_local(gemm->B_data_gpu.data->memory_type_index);
            d::Json layers=d::Json::array();
            {ncnn::VkCompute cmd(device);
                auto outputs=d::graph_forward(net,std::vector<ncnn::VkMat>{current},&cmd,runtime_option,layers);
                require(!cmd.submit_and_wait(),"GEMM execution failed");current=outputs.at(0);
            }
            ncnn::Mat output;
            {ncnn::VkCompute cmd(device);cmd.record_download(current,output,runtime_option);require(!cmd.submit_and_wait(),"Download failed");}
            // Exact powers of two/three give an oracle independent of ncnn.
            const float expected0=1.25f*float(1u<<(step+1));
            const std::array<float,4> expected1{-6.f,-18.f,-54.f,-162.f};
            if(output.w*output.h*output.d*output.c*output.elempack!=2||output[0]!=expected0||output[1]!=expected1[step]) {
                std::cerr<<"GEMM step "<<step<<" shape "<<d::Json(d::tensor_shape(output)).dump()<<" pack "<<output.elempack
                    <<" actual "<<output[0]<<','<<output[1]<<" expected "<<expected0<<','<<expected1[step]<<'\n';
                throw std::runtime_error("GEMM oracle differs");
            }
            require(layers.size()==1&&layers[0]["backend"]=="vulkan","CPU fallback");
            placement["exact_output"]=true;results.push_back(placement);
            // current remains valid after net destruction, and is reused by the
            // next graph. This exercises completion-before-release ownership.
        }
        require(queries==4,"Budget was not refreshed for every graph");
        {
            ncnn::Net net;net.opt=option;net.set_vulkan_device(0);
            bool rejected=false;
            try{d::configure_memory(net,temporary.path/"absent.bin",{});}catch(const std::exception &){rejected=true;}
            require(rejected,"Missing weights were accepted");
        }
        std::cout<<d::Json{{"passed",true},{"scope","Four real Vulkan GEMMs; controlled budgets are not physical VRAM exhaustion"},
            {"device",d::device_info(0)},
            {"live_driver_budget",d::budget_json(d::device_budget(device))},{"steps",results}}.dump(2)<<'\n';
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
