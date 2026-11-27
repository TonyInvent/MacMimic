#pragma once
#include <onnxruntime_cxx_api.h>
#include "policy_core.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sprite_policy {
class StandingGroove {
    static Ort::SessionOptions options() {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        return opts;
    }
    Ort::Session session_;
    Ort::MemoryInfo memory_=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
public:
    const int frames;
    StandingGroove(Ort::Env& env,const char* path,int count):session_(env,path,options()),frames(count) {
        Ort::AllocatorWithDefaultOptions allocator;
        auto check=[](const Ort::TypeInfo& info,ONNXTensorElementDataType type,std::vector<int64_t> shape) {
            auto tensor=info.GetTensorTypeAndShapeInfo();
            return tensor.GetElementType()==type && tensor.GetShape()==shape;
        };
        if(session_.GetInputCount()!=2 || session_.GetOutputCount()!=1 ||
           std::string(session_.GetInputNameAllocated(0,allocator).get())!="history" ||
           std::string(session_.GetInputNameAllocated(1,allocator).get())!="frame" ||
           std::string(session_.GetOutputNameAllocated(0,allocator).get())!="actions" ||
           !check(session_.GetInputTypeInfo(0),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,{1,10,99}) ||
           !check(session_.GetInputTypeInfo(1),ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,{1}) ||
           !check(session_.GetOutputTypeInfo(0),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,{1,31}))
            throw std::runtime_error("Standing groove model signature mismatch");
    }
    void infer(const std::array<float,993>& packed,int64_t frame,std::array<float,31>& raw) {
        if(frame<0 || frame>=frames)throw std::runtime_error("Standing groove frame outside choreography");
        std::array<float,990> history;
        for(int t=0;t<10;++t) {
            auto dst=history.begin()+99*t;
            std::copy_n(packed.begin()+31*t,31,dst);
            std::copy_n(packed.begin()+310+31*t,31,dst+31);
            std::copy_n(packed.begin()+930+3*t,3,dst+62);
            std::copy_n(packed.begin()+960+3*t,3,dst+65);
            std::copy_n(packed.begin()+620+31*t,31,dst+68);
        }
        const int64_t hs[]={1,10,99}, fs[]={1}, os[]={1,31};
        Ort::Value inputs[]={Ort::Value::CreateTensor<float>(memory_,history.data(),history.size(),hs,3),
                             Ort::Value::CreateTensor<int64_t>(memory_,&frame,1,fs,1)};
        auto output=Ort::Value::CreateTensor<float>(memory_,raw.data(),raw.size(),os,2);
        const char* in[]={"history","frame"}, *out[]={"actions"};
        session_.Run(Ort::RunOptions{nullptr},in,inputs,2,out,&output,1);
        for(float x:raw)if(!std::isfinite(x))throw std::runtime_error("Nonfinite groove action");
    }
};

inline void replay_groove(const char* model,const char* fixture,const char* result,bool feedback,int frames) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"standing-groove-replay");
    StandingGroove groove(env,model,frames);
    std::ifstream input(fixture,std::ios::binary);
    std::ofstream output(result,std::ios::binary);
    auto read=[&](auto& array) {
        if(!input.read(reinterpret_cast<char*>(array.data()),sizeof(array)))throw std::runtime_error("Incomplete groove fixture");
    };
    SuperMimicCore core;
    if(feedback) {
        read(core.defaults);
        read(core.scale);
        core.offset=core.defaults;
    }
    for(int i=0;i<frames;++i) {
        std::array<float,993> packed;
        std::array<float,31> raw;
        std::array<double,71> state{};
        if(feedback) {
            read(state);
            packed=core.observe(state);
        } else read(packed);
        groove.infer(packed,i,raw);
        if(feedback)core.latch(raw,state,true);
        if(!output.write(reinterpret_cast<const char*>(raw.data()),sizeof(raw)))throw std::runtime_error("Cannot write groove replay");
    }
    if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing groove fixture bytes");
    output.close();
    if(!output)throw std::runtime_error("Cannot flush groove replay");
}
}
