//
//  inference.hpp
//  detection
//
//  Created by yang on 2020/12/4.
//

#ifndef inference_hpp
#define inference_hpp

#include <vector>


enum PrecisionType {
    kFp32,
    kFp16,
    kInt8
};


enum ForwardType {
    kForwardCpu = 0,
    /*Hand write metal*/
    kForwardMetal = 1,
};


typedef struct _InitParam {
    int num_threads = 4;
    PrecisionType precision_type = kFp32;
    ForwardType forward_type = kForwardCpu;
} InitParam;


class ISession {
    public:
        virtual ~ISession() {};
        virtual std::vector<float*> run(float* input_data) = 0;
};


#ifdef __cplusplus
extern "C" {
#endif

ISession *CreateSession(const char *model_path, InitParam param);
void  ReleaseSession(ISession **session_ptr);                      // ReleaseSession(&modelSession);

#ifdef __cplusplus
}
#endif


#endif /* inference_hpp */
