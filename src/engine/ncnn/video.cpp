#include "inference.hpp"
#include "provenance.hpp"
#include "video_io.hpp"
#include "seedvr2/video.hpp"
#include <bit>

namespace seedvr2::engine {
using namespace detail;
using namespace inference;
Result<std::string> inspect_video(const std::filesystem::path &path) {
    try {
        const auto clip=load_video(path,1,{},true);
        return Json{{"width",clip.width},{"height",clip.height},{"fps",clip.fps},
            {"duration_seconds",clip.duration_seconds},{"has_audio",clip.has_audio},
            {"sha256",hash(path)},{"bytes",std::filesystem::file_size(path)},{"kind","video"}}.dump();
    } catch (const std::exception &e) { return Error{"VIDEO_INVALID","input",e.what()}; }
}
Result<std::string> inspect_video_package(const std::filesystem::path &directory) {
    try {
        Package p(directory,{},true);
        return Json{{"status","INTEGRITY_CHECKED"},{"profile",p.manifest.at("profile")},{"manifest_sha256",p.identity},
            {"graphs",p.graphs.size()},{"limits",p.manifest.at("limits")},{"model_verified",false}}.dump();
    } catch (const std::exception &e) { return Error{"MODEL_PACKAGE_INVALID","model",e.what()}; }
}
Result<std::string> run_video(const VideoRequest &request,ImageObserver observer,CancellationCheck cancelled) {
    try {
        const auto started=Clock::now();
        const auto ready=seedvr2::preflight(public_request(request,true,request.max_frames));
        if (is_error(ready)) return std::get<Error>(ready);
        if constexpr (std::endian::native!=std::endian::little) throw std::runtime_error("FP32 package needs a little endian host");
        if (request.max_frames<1 || request.max_frames>17 || request.long_side<64 || request.long_side>128 ||
            request.long_side%16 || request.threads<1 || request.threads>32 || request.gpu_index< -1)
            throw std::runtime_error("Short-video preview supports 1..17 frames and 64..128 output long side, divisible by 16");
        std::lock_guard lock(engine_mutex);
        auto check=[&] {if (cancelled && cancelled()) throw Cancelled();};
        auto progress=[&](std::string stage,int completed) {
            check();if (observer) observer({std::move(stage),completed,38,std::chrono::duration<double,std::milli>(Clock::now()-started).count()});
        };
        progress("validating",0);
        if (std::filesystem::exists(request.output_directory) && !std::filesystem::is_empty(request.output_directory))
            throw std::runtime_error("Output directory must be empty");
        const auto input_hash=hash(request.input_file);
        auto clip=load_video(request.input_file,request.max_frames,cancelled);
        const int frames=static_cast<int>(clip.frames.size()), padded=((frames-1+3)/4)*4+1, lt=(padded-1)/4+1;
        auto first=prepare_image(clip.frames.front(),request.long_side);
        const int h=first.h,w=first.w,lh=h/8,lw=w/8,gh=h/16,gw=w/16,tokens=lt*gh*gw;
        ncnn::Mat prepared(w,h,padded,3,size_t(4),1);
        if (prepared.empty()) throw std::runtime_error("Cannot allocate video tensor");
        for (int t=0;t<padded;++t) {
            check();const auto frame=prepare_image(clip.frames[std::min(t,frames-1)],request.long_side);
            for (int c=0;c<3;++c) std::memcpy(static_cast<float *>(prepared.channel(c))+t*h*w,frame.channel(c).data,std::size_t(h)*w*4);
        }
        const auto validation_started=Clock::now();
        Package package(request.model_directory,check,true,[&](const Progress &) { progress("validating",0); });
        const auto validation_ms=std::chrono::duration<double,std::milli>(Clock::now()-validation_started).count();
        int gpu=-1;Json gpu_info=nullptr;std::unique_ptr<ncnn::PipelineCache> pipelines;
        if (request.vulkan) {
            init_gpu();gpu=request.gpu_index<0?ncnn::get_default_gpu_index():request.gpu_index;
            if (gpu<0 || gpu>=ncnn::get_gpu_count()) throw std::runtime_error("Requested Vulkan device is unavailable");
            gpu_info=device_info(gpu);pipelines=std::make_unique<ncnn::PipelineCache>(ncnn::get_gpu_device(gpu));
        }
        std::filesystem::create_directories(request.output_directory);
        Json diagnostics=Json::object(),stages=Json::array();
        auto capture=[&](const std::string &name,const ncnn::Mat &value) {
            if (request.diagnostic_tensors) {
                auto row=write_tensor(request.output_directory,(name+".f32").c_str(),value);
                row["shape"]=tensor_shape(value);diagnostics[name]=row;
            }
        };
        auto execute=[&](const std::string &name,const std::vector<ncnn::Mat> &in) {
            return execute_graph(package.graphs.at(name),name,in,request,gpu,stages,check,pipelines.get());
        };
        capture("prepared",prepared);
        save_video(request.output_directory/"comparison-input.mp4",prepared,clip,cancelled);
        save_image(request.output_directory/"comparison-input.png",first);
        progress("encoding",1);
        auto posterior=execute("encoder",{prepared})[0];expect(posterior,{32,lt,lh,lw},"video encoder");
        capture("posterior",posterior);prepared.release();
        progress("sampling",2);
        ncnn::Mat posterior_noise(lw,lh,lt,16,size_t(4),1),noise(lw,lh,lt,16,size_t(4),1),conditioned(lw,lh,lt,16,size_t(4),1);
        NormalNoise random(request.seed);
        for (int c=0;c<16;++c) {
            auto p=posterior_noise.channel(c),z=conditioned.channel(c),mean=posterior.channel(c),logvar=posterior.channel(c+16);
            for (int j=0;j<lt*lh*lw;++j) {p[j]=random.next();z[j]=(mean[j]+std::exp(.5f*std::clamp(logvar[j],-30.f,20.f))*p[j])*.9152f;}
        }
        for (int c=0;c<16;++c) {auto n=noise.channel(c);for (int j=0;j<lt*lh*lw;++j) n[j]=random.next();}
        capture("posterior-noise",posterior_noise);capture("noise",noise);capture("conditioned",conditioned);
        posterior.release();posterior_noise.release();
        ncnn::Mat patches(132,tokens,size_t(4),1);
        for (int t=0;t<lt;++t) for (int y=0;y<gh;++y) for (int x=0;x<gw;++x) {
            auto *p=patches.row((t*gh+y)*gw+x);
            for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx) {
                const int offset=(dy*2+dx)*33,index=(t*lh+2*y+dy)*lw+2*x+dx;
                for (int c=0;c<16;++c) {p[offset+c]=noise.channel(c)[index];p[offset+c+16]=conditioned.channel(c)[index];}
                p[offset+32]=1.f;
            }
        }
        capture("patches",patches);conditioned.release();progress("projecting",3);
        auto video=execute("patch-in",{patches})[0];expect(video,{tokens,2560},"patch-in");
        capture("patch-in",video);patches.release();
        auto text=package.text;capture("text-in",text);capture("time-in",package.time);
        video=video.reshape(2560,gw,gh,lt);
        for (int i=0;i<32;++i) {
            const auto name="block-"+std::string(i<10?"0":"")+std::to_string(i);
            progress(name,4+i);auto out=execute(name,{video,text,package.time});video=out[0];text=out[1];
            expect(video,{lt,gh,gw,2560},name);expect(text,{58,2560},name+" text");
            capture(name+"-video",video.reshape(2560,tokens));capture(name+"-text",text);
        }
        progress("denoising",36);
        auto prediction=execute("patch-out",{video.reshape(2560,tokens)})[0];expect(prediction,{tokens,64},"patch-out");capture("patch-out",prediction);
        ncnn::Mat velocity(lw,lh,lt,16,size_t(4),1),latent(lw,lh,lt,16,size_t(4),1);
        for (int t=0;t<lt;++t) for (int y=0;y<gh;++y) for (int x=0;x<gw;++x)
            for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx) for (int c=0;c<16;++c) {
                const auto value=prediction.row((t*gh+y)*gw+x)[(dy*2+dx)*16+c];
                const int index=(t*lh+2*y+dy)*lw+2*x+dx;
                velocity.channel(c)[index]=value;latent.channel(c)[index]=(noise.channel(c)[index]-value)/.9152f;
            }
        capture("velocity",velocity);capture("latent",latent);
        video.release();text.release();prediction.release();noise.release();velocity.release();
        progress("decoding",37);
        auto decoded=execute("decoder",{latent})[0];expect(decoded,{3,padded,h,w},"video decoder");capture("decoded",decoded);
        check();save_video(request.output_directory/"output.partial.mp4",decoded,clip,cancelled);
        ncnn::Mat poster(w,h,3,size_t(4),1);
        for (int c=0;c<3;++c) std::memcpy(poster.channel(c).data,decoded.channel(c).data,std::size_t(w)*h*4);
        save_image(request.output_directory/"output.png",poster);
        if (hash(request.input_file)!=input_hash || hash(request.model_directory/"manifest.json")!=package.identity)
            throw std::runtime_error("Input or model manifest changed during execution");
        check();
        Json report{{"schema_version","seedvr2-video-run-v1"},{"status","SUCCEEDED"},{"build",SEEDVR2_BUILD_VERSION},
            {"profile",package.manifest.at("profile")},{"backend",request.vulkan?"ncnn-vulkan":"ncnn-cpu"},{"device",gpu_info},
            {"ncnn_commit",SEEDVR2_NCNN_COMMIT},{"model_manifest_sha256",package.identity},{"model_verified",false},
            {"numerical_validation","NOT_PERFORMED_BY_RUNNER"},{"model_certificate",nullptr},
            {"input",{{"sha256",input_hash},{"width",clip.width},{"height",clip.height},{"duration_seconds",clip.duration_seconds},{"has_audio",clip.has_audio}}},
            {"output",{{"path","output.mp4"},{"sha256",hash(request.output_directory/"output.partial.mp4")},{"width",w},{"height",h},
                {"frames",frames},{"fps",clip.fps},{"timestamps_90khz",clip.timestamps},{"audio",false},{"codec","H.264 / yuv420p / CRF 18"}}},
            {"clip",{{"requested_max_frames",request.max_frames},{"decoded_frames",frames},{"padded_frames",padded},{"latent_frames",lt},
                {"truncated",clip.truncated},{"padding","repeat last frame to 4n+1; crop output to decoded frame count"},
                {"temporal_mode","joint clip VAE + 3D adaptive window attention"},{"streaming_cache",false}}},
            {"comparison_input","comparison-input.mp4"},{"poster","output.png"},{"seed",request.seed},
            {"noise_algorithm","splitmix64-box-muller-v1"},{"sampling",package.manifest.at("sampling")},
            {"preprocessing","FFmpeg RGB8 / bicubic antialias / center crop 16 / normalize [-1,1]"},
            {"media_version",media_version()},{"host_operations",{"video codecs","resizing","layouts","posterior sampling","Euler endpoint"}},
            {"dispatch","EXPLICIT_PER_LAYER_NO_BACKEND_FALLBACK"},{"stages",stages},{"diagnostics",diagnostics},
            {"total_ms",std::chrono::duration<double,std::milli>(Clock::now()-started).count()}};
        report["storage_precision"] = package.manifest.value("storage_precision", Json{{"dit_linear_weights","fp32"},{"other_weights","fp32"},{"activation","fp32"},{"arithmetic","fp32"}});
        report["implementation"]=implementation_identity();
        if (report["implementation"].contains("executable_sha256")) report["executable_sha256"]=report["implementation"]["executable_sha256"];
        report["resources"]=process_resources();
        report["resources"]["weight_placement"]=memory_summary(stages);
        report["weight_io"]=request.mapped_weights?"mapped":"buffered";
        const auto &preflight=std::get<Preflight>(ready);
        report["resources"]["package_bytes"]=preflight.package_bytes;
        report["resources"]["largest_graph_weight_bytes"]=preflight.largest_graph_bytes;
        double loads=0,compute=0;
        for (const auto &stage:stages) {loads+=stage.at("load_ms").get<double>();compute+=stage.at("compute_ms").get<double>();}
        report["timing"]={{"package_validation_ms",validation_ms},{"graph_load_ms",loads},{"graph_compute_ms",compute},
            {"other_host_wait_and_cleanup_ms",report.at("total_ms").get<double>()-validation_ms-loads-compute}};
        report["package_authenticated"]=true;
        report["converter_ncnn_commit"]=package.manifest.at("ncnn_commit");

        const auto result=report.dump(2);std::ofstream f(request.output_directory/"run.partial.json");f<<result<<'\n';f.close();
        if (!f) throw std::runtime_error("Cannot write video execution report");
        check();std::filesystem::rename(request.output_directory/"output.partial.mp4",request.output_directory/"output.mp4");
        std::filesystem::rename(request.output_directory/"run.partial.json",request.output_directory/"run.json");
        progress("completed",38);return result;
    } catch (const Cancelled &e) { return Error{"CANCELLED","run",e.what()}; }
      catch (const std::exception &e) { return Error{"VIDEO_EXECUTION_FAILED","run",e.what()}; }
}
}
