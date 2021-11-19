//
//  inference.cpp
//  detection
//
//  Created by yang on 2020/12/4.
//

#include "inference.hpp"

#include <atomic>
#include <string>
#include <iostream>
#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>


class Session: public ISession {
    public:
        Session(const char* model_path, int num_thread, ForwardType forward_type, PrecisionType precision_type);
        std::vector<float*> run(float *input_data) override;
        ~Session();
    
    private:
        MNN::Interpreter *m_net;
        MNN::Session *m_session;
        std::vector<MNN::Tensor*> m_tensor;
        void clearTensor();                     // 清除 Tensor 缓存
        
        static std::atomic<int> num_session;
};

std::atomic<int> Session::num_session(0);

void Session::clearTensor(){
    if (!m_tensor.empty()) {
        for (auto &tensor : m_tensor){
            delete tensor;
            tensor = nullptr;
        }
        m_tensor.clear();
    }
}

Session::Session(const char* model_path, int num_thread, ForwardType forward_type, PrecisionType precision_type){
    m_net = MNN::Interpreter::createFromFile(model_path);
    if (m_net == nullptr) {
        std::cerr << "create interpreter failed ! " << std::endl;
    }
    
    num_session += 1;                           // 计数加一
    
    MNN::ScheduleConfig config;
    config.numThread  = num_thread;
    config.type       = (MNNForwardType)forward_type;
    
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (precision_type == kFp32) ? MNN::BackendConfig::Precision_High : MNN::BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    
    m_session = m_net->createSession(config);
    if (m_session == nullptr){
        std::cerr << "create session failed ! " << std::endl;
    }
}

std::vector<float*> Session::run(float *input_data) {
    std::vector<float*> feature_map_ptrs;
    clearTensor();                          // clear tensor buffer
    
    MNN::Tensor* input_tensor = m_net->getSessionInput(m_session, nullptr);
    if (input_tensor == nullptr) {
        std::cerr << "get session input failed ! " << std::endl;
    }
    
    auto dim_type = input_tensor->getDimensionType();
    MNN::Tensor given_tensor(input_tensor, dim_type);
    auto tensor_input_data = given_tensor.host<float>();
    auto size = given_tensor.elementSize();
    memcpy(tensor_input_data, input_data, size * sizeof(float));
    input_tensor->copyFromHostTensor(&given_tensor);
    
    m_net->runSession(m_session);
    
    // output
    std::map<std::string, MNN::Tensor*> output_tensors = m_net->getSessionOutputAll(m_session);
    for (auto iter = output_tensors.begin(); iter != output_tensors.end(); iter++) {
        MNN::Tensor* output_tensor = iter->second;
        auto nchw_tensor = new MNN::Tensor(output_tensor, output_tensor->getDimensionType());
        if(nchw_tensor == nullptr) {
            std::cerr << "create tensor failed ! " << std::endl;
            continue;
        }
        
        output_tensor->copyToHostTensor(nchw_tensor);
        feature_map_ptrs.push_back(nchw_tensor->host<float>());
        m_tensor.push_back(nchw_tensor);
    }
    return feature_map_ptrs;
}

Session::~Session() {
    clearTensor();
    
    if (m_net) {
        if (m_session) {
            m_net->releaseSession(m_session);
            m_session = nullptr;
        }
        delete m_net;
        m_net = nullptr;
    }
    
    num_session -= 1;                   // 计数减一
};

ISession *CreateSession(const char *model_path, InitParam param) {
    return new Session(model_path, param.num_threads, param.forward_type, param.precision_type);
}

void ReleaseSession(ISession **session_ptr) {
    if (session_ptr && *session_ptr) {
        delete *session_ptr;
        session_ptr = nullptr;
    }
}
