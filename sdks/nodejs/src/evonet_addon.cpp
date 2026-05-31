#include <node_api.h>
#include <string>

// Giả lập thư viện libevonet C++ của framework
namespace evonet {
    std::string infer_mamba(const std::string& prompt) {
        // Thực tế sẽ gọi tensor operations của PyTorch C++ API (libtorch) ở đây
        return "[C++ Native] Xử lý với độ trễ 0ms cho prompt: " + prompt;
    }
}

// Hàm Wrapper N-API để Node.js có thể gọi
napi_value InferMambaWrapped(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    char prompt[1024];
    size_t prompt_len;
    napi_get_value_string_utf8(env, args[0], prompt, sizeof(prompt), &prompt_len);

    // Gọi lõi C++ Native
    std::string result = evonet::infer_mamba(std::string(prompt));

    // Chuyển kết quả C++ về biến String của JavaScript
    napi_value js_result;
    napi_create_string_utf8(env, result.c_str(), result.length(), &js_result);

    return js_result;
}

// Khởi tạo và Đăng ký module với Node.js V8 Engine
napi_value Init(napi_env env, napi_value exports) {
    napi_value fn;
    napi_create_function(env, nullptr, 0, InferMambaWrapped, nullptr, &fn);
    napi_set_named_property(env, exports, "infer", fn);
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
