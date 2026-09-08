#include "video_io.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
namespace seedvr2::engine::detail {
namespace {
void require(int result, const char *message) {
    if (result < 0) { char error[AV_ERROR_MAX_STRING_SIZE]; av_strerror(result,error,sizeof(error));throw std::runtime_error(std::string(message)+": "+error); }
}
void check_color(int transfer, int primaries, int space, int format) {
    if (transfer==AVCOL_TRC_SMPTE2084 || transfer==AVCOL_TRC_ARIB_STD_B67 ||
        primaries==AVCOL_PRI_BT2020 || space==AVCOL_SPC_BT2020_NCL || space==AVCOL_SPC_BT2020_CL)
        throw std::runtime_error("HDR and BT.2020 video need tone mapping before this SDR preview");
    if (const auto *desc=av_pix_fmt_desc_get(static_cast<AVPixelFormat>(format)))
        for (int c=0;c<desc->nb_components;++c) if (desc->comp[c].depth>8)
            throw std::runtime_error("This video preview accepts 8-bit SDR pixels only");
}
int color_matrix(int space) {
    switch (space) {
    case AVCOL_SPC_BT709: return SWS_CS_ITU709;
    case AVCOL_SPC_FCC: return SWS_CS_FCC;
    case AVCOL_SPC_SMPTE240M: return SWS_CS_SMPTE240M;
    case AVCOL_SPC_RGB: case AVCOL_SPC_UNSPECIFIED:
    case AVCOL_SPC_BT470BG: case AVCOL_SPC_SMPTE170M: return SWS_CS_ITU601;
    default: throw std::runtime_error("Unsupported input video color matrix");
    }
}
struct Input {
    AVFormatContext *format=nullptr;AVCodecContext *codec=nullptr;AVFrame *frame=av_frame_alloc();
    AVPacket *packet=av_packet_alloc();SwsContext *scale=nullptr;
    std::function<bool()> cancelled;
    std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    static int interrupt(void *opaque) {
        const auto &s=*static_cast<Input *>(opaque);
        return (s.cancelled && s.cancelled()) || std::chrono::steady_clock::now()-s.started>std::chrono::seconds(30);
    }
    ~Input() { sws_freeContext(scale);av_packet_free(&packet);av_frame_free(&frame);avcodec_free_context(&codec);avformat_close_input(&format); }
};
struct Output {
    AVFormatContext *format=nullptr;AVCodecContext *codec=nullptr;AVFrame *frame=av_frame_alloc();
    AVPacket *packet=av_packet_alloc();SwsContext *scale=nullptr;
    ~Output() { sws_freeContext(scale);av_packet_free(&packet);av_frame_free(&frame);avcodec_free_context(&codec);
        if (format) { if (format->pb) avio_closep(&format->pb);avformat_free_context(format); } }
};
}
std::string media_version() { return av_version_info(); }
VideoClip load_video(const std::filesystem::path &path, int limit, const std::function<bool()> &cancelled, bool probe) {
    if (limit<1 || limit>17 || std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path)>256ULL*1024*1024) throw std::runtime_error("Video must be a local MP4/WebM/MKV file up to 256 MiB");
    Input s;s.cancelled=cancelled;s.format=avformat_alloc_context();
    if (!s.frame || !s.packet || !s.format) throw std::runtime_error("Cannot allocate video decoder");
    s.format->interrupt_callback={Input::interrupt,&s};
    AVDictionary *options=nullptr;
    av_dict_set(&options,"protocol_whitelist","file",0);
    av_dict_set(&options,"format_whitelist","mov,matroska,webm",0);
    av_dict_set(&options,"probesize","1048576",0);
    av_dict_set(&options,"analyzeduration","2000000",0);
    int status=avformat_open_input(&s.format,std::filesystem::absolute(path).c_str(),nullptr,&options);
    av_dict_free(&options);require(status,"Cannot open video");
    require(avformat_find_stream_info(s.format,nullptr),"Cannot inspect video");
    const AVCodec *decoder=nullptr;
    const int index=av_find_best_stream(s.format,AVMEDIA_TYPE_VIDEO,-1,-1,&decoder,0);
    require(index,"Video stream missing");
    auto *stream=s.format->streams[index];const auto *par=stream->codecpar;
    check_color(par->color_trc,par->color_primaries,par->color_space,par->format);
    color_matrix(par->color_space);
    if (par->width<16 || par->height<16 || par->width>4096 || par->height>4096 ||
        std::int64_t(par->width)*par->height>4096*2160) throw std::runtime_error("Video dimensions exceed the preview import limit");
    const auto sar=stream->sample_aspect_ratio.num?stream->sample_aspect_ratio:par->sample_aspect_ratio;
    if (sar.num && sar.den && sar.num!=sar.den) throw std::runtime_error("Non-square video pixels are not supported in this preview");
    const auto *rotation=av_packet_side_data_get(par->coded_side_data,par->nb_coded_side_data,AV_PKT_DATA_DISPLAYMATRIX);
    if (rotation && rotation->size>=9*sizeof(std::int32_t) &&
        std::abs(av_display_rotation_get(reinterpret_cast<const std::int32_t *>(rotation->data)))>.01)
        throw std::runtime_error("Rotate this video to upright pixels before importing");
    const AVRational rate=av_guess_frame_rate(s.format,stream,nullptr);
    VideoClip clip;clip.width=par->width;clip.height=par->height;clip.fps=av_q2d(rate);
    if (!std::isfinite(clip.fps) || clip.fps<1 || clip.fps>240) throw std::runtime_error("Video frame rate must be 1..240 fps");
    clip.duration_seconds=stream->duration!=AV_NOPTS_VALUE?double(stream->duration)*av_q2d(stream->time_base):
        (s.format->duration!=AV_NOPTS_VALUE?double(s.format->duration)/AV_TIME_BASE:0.);
    for (unsigned i=0;i<s.format->nb_streams;++i) clip.has_audio|=s.format->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_AUDIO;
    if (probe) return clip;
    s.codec=avcodec_alloc_context3(decoder);if (!s.codec) throw std::runtime_error("Cannot allocate video codec");
    require(avcodec_parameters_to_context(s.codec,par),"Video parameters invalid");
    s.codec->thread_count=4;
    require(avcodec_open2(s.codec,decoder,nullptr),"Cannot open video decoder");
    std::int64_t first=AV_NOPTS_VALUE;
    auto receive=[&] {
        for (;;) {
            const int code=avcodec_receive_frame(s.codec,s.frame);
            if (code==AVERROR(EAGAIN)||code==AVERROR_EOF) break;
            require(code,"Video decode failed");
            if (Input::interrupt(&s)) throw std::runtime_error("Video decode cancelled or timed out");
            if (clip.frames.size()==static_cast<std::size_t>(limit)) {clip.truncated=true;av_frame_unref(s.frame);return;}
            if (s.frame->width!=clip.width || s.frame->height!=clip.height) throw std::runtime_error("Changing video dimensions are not supported");
            check_color(s.frame->color_trc,s.frame->color_primaries,s.frame->colorspace,s.frame->format);
            s.scale=sws_getCachedContext(s.scale,clip.width,clip.height,static_cast<AVPixelFormat>(s.frame->format),
                clip.width,clip.height,AV_PIX_FMT_RGB24,SWS_BICUBIC,nullptr,nullptr,nullptr);
            if (!s.scale) throw std::runtime_error("Video color conversion unavailable");
            const auto space=s.frame->colorspace==AVCOL_SPC_UNSPECIFIED?par->color_space:s.frame->colorspace;
            const auto range=s.frame->color_range==AVCOL_RANGE_UNSPECIFIED?par->color_range:s.frame->color_range;
            const auto *desc=av_pix_fmt_desc_get(static_cast<AVPixelFormat>(s.frame->format));
            const bool full=range==AVCOL_RANGE_JPEG || (range==AVCOL_RANGE_UNSPECIFIED && desc &&
                ((desc->flags&AV_PIX_FMT_FLAG_RGB) || std::string_view(desc->name).starts_with("yuvj")));
            const auto *coeff=sws_getCoefficients(color_matrix(space));
            require(sws_setColorspaceDetails(s.scale,coeff,full,coeff,1,0,1<<16,1<<16),"Cannot set input color matrix");
            ImagePixels image{clip.width,clip.height,std::vector<unsigned char>(std::size_t(clip.width)*clip.height*3)};
            std::uint8_t *dst[]={image.rgb.data()};int stride[]={clip.width*3};
            require(sws_scale(s.scale,s.frame->data,s.frame->linesize,0,clip.height,dst,stride),"Video pixel conversion failed");
            auto stamp=s.frame->best_effort_timestamp;
            if (first==AV_NOPTS_VALUE && stamp!=AV_NOPTS_VALUE) first=stamp;
            stamp=(stamp!=AV_NOPTS_VALUE && first!=AV_NOPTS_VALUE)?av_rescale_q(stamp-first,stream->time_base,{1,90000}):
                static_cast<std::int64_t>(std::llround(double(clip.frames.size())*90000./clip.fps));
            if (stamp<0 || (!clip.timestamps.empty() && stamp<=clip.timestamps.back())) throw std::runtime_error("Video timestamps are not increasing");
            clip.timestamps.push_back(stamp);clip.frames.push_back(std::move(image));av_frame_unref(s.frame);
        }
    };
    while (!clip.truncated) {
        status=av_read_frame(s.format,s.packet);
        if (status==AVERROR_EOF) break;
        require(status,"Cannot read video packet");
        if (s.packet->stream_index==index) { require(avcodec_send_packet(s.codec,s.packet),"Cannot send video packet");receive(); }
        av_packet_unref(s.packet);
    }
    if (!clip.truncated) { require(avcodec_send_packet(s.codec,nullptr),"Cannot flush video decoder");receive(); }
    if (clip.frames.empty()) throw std::runtime_error("Video has no decodable frames");
    return clip;
}
void save_video(const std::filesystem::path &path,const ncnn::Mat &pixels,const VideoClip &clip,const std::function<bool()> &cancelled) {
    if (pixels.dims!=4 || pixels.c!=3 || pixels.elempack!=1 || pixels.elemsize!=4 ||
        pixels.d<static_cast<int>(clip.frames.size()) || clip.frames.size()!=clip.timestamps.size() || clip.frames.empty())
        throw std::runtime_error("Invalid video output tensor");
    Output s;require(avformat_alloc_output_context2(&s.format,nullptr,"mp4",path.c_str()),"Cannot create MP4");
    const auto *encoder=avcodec_find_encoder_by_name("libx264");
    if (!encoder || !s.format || !s.frame || !s.packet) throw std::runtime_error("This FFmpeg build needs the libx264 encoder for browser playback");
    auto *stream=avformat_new_stream(s.format,nullptr);s.codec=avcodec_alloc_context3(encoder);
    if (!stream || !s.codec) throw std::runtime_error("Cannot allocate video encoder");
    s.codec->width=pixels.w;s.codec->height=pixels.h;s.codec->pix_fmt=AV_PIX_FMT_YUV420P;
    s.codec->time_base={1,90000};s.codec->framerate=av_d2q(clip.fps,100000);s.codec->thread_count=4;s.codec->max_b_frames=0;
    s.codec->color_primaries=AVCOL_PRI_BT709;s.codec->color_trc=AVCOL_TRC_BT709;
    s.codec->colorspace=AVCOL_SPC_BT709;s.codec->color_range=AVCOL_RANGE_MPEG;
    if (s.format->oformat->flags&AVFMT_GLOBALHEADER) s.codec->flags|=AV_CODEC_FLAG_GLOBAL_HEADER;
    AVDictionary *options=nullptr;av_dict_set(&options,"preset","fast",0);av_dict_set(&options,"crf","18",0);
    int code=avcodec_open2(s.codec,encoder,&options);av_dict_free(&options);require(code,"Cannot open H.264 encoder");
    require(avcodec_parameters_from_context(stream->codecpar,s.codec),"Cannot set output parameters");
    stream->time_base=s.codec->time_base;stream->avg_frame_rate=s.codec->framerate;
    require(avio_open(&s.format->pb,path.c_str(),AVIO_FLAG_WRITE),"Cannot write output video");
    av_dict_set(&options,"movflags","+faststart",0);code=avformat_write_header(s.format,&options);av_dict_free(&options);require(code,"Cannot write video header");
    s.frame->format=s.codec->pix_fmt;s.frame->width=pixels.w;s.frame->height=pixels.h;
    require(av_frame_get_buffer(s.frame,32),"Cannot allocate output frame");
    s.scale=sws_getContext(pixels.w,pixels.h,AV_PIX_FMT_RGB24,pixels.w,pixels.h,s.codec->pix_fmt,SWS_BICUBIC,nullptr,nullptr,nullptr);
    if (!s.scale) throw std::runtime_error("Cannot create video color converter");
    require(sws_setColorspaceDetails(s.scale,sws_getCoefficients(SWS_CS_ITU709),1,sws_getCoefficients(SWS_CS_ITU709),0,0,1<<16,1<<16),"Cannot set output color matrix");
    std::vector<std::uint8_t> rgb(std::size_t(pixels.w)*pixels.h*3);
    auto drain=[&] {
        for (;;) {
            const int r=avcodec_receive_packet(s.codec,s.packet);
            if (r==AVERROR(EAGAIN)||r==AVERROR_EOF) break;
            require(r,"H.264 encoding failed");
            av_packet_rescale_ts(s.packet,s.codec->time_base,stream->time_base);s.packet->stream_index=stream->index;
            require(av_interleaved_write_frame(s.format,s.packet),"MP4 write failed");av_packet_unref(s.packet);
        }
    };
    for (std::size_t t=0;t<clip.frames.size();++t) {
        if (cancelled && cancelled()) throw std::runtime_error("Video encoding cancelled");
        require(av_frame_make_writable(s.frame),"Cannot write output frame");
        for (int c=0;c<3;++c) {
            const auto *src=static_cast<const float *>(pixels.channel(c))+t*std::size_t(pixels.w)*pixels.h;
            for (int i=0;i<pixels.w*pixels.h;++i) {
                if (!std::isfinite(src[i])) throw std::runtime_error("Non-finite video output");
                rgb[std::size_t(i)*3+c]=static_cast<std::uint8_t>(std::lround((std::clamp(src[i],-1.f,1.f)*.5f+.5f)*255.f));
            }
        }
        const std::uint8_t *src[]={rgb.data()};int stride[]={pixels.w*3};
        require(sws_scale(s.scale,src,stride,0,pixels.h,s.frame->data,s.frame->linesize),"Cannot convert output pixels");
        s.frame->pts=clip.timestamps[t];
        s.frame->duration=t+1<clip.timestamps.size()?clip.timestamps[t+1]-clip.timestamps[t]:static_cast<std::int64_t>(std::llround(90000./clip.fps));
        require(avcodec_send_frame(s.codec,s.frame),"Cannot send output frame");drain();
    }
    require(avcodec_send_frame(s.codec,nullptr),"Cannot flush output video");drain();
    require(av_write_trailer(s.format),"Cannot finalize output video");
}
}
