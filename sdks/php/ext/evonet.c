#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "php_ini.h"
#include "ext/standard/info.h"

// Hàm C nội bộ (Sẽ liên kết vào libevonet.so C++)
char* infer_native(char* prompt) {
    // Mock C++ Native call
    char* result = (char*)emalloc(strlen(prompt) + 100);
    sprintf(result, "[PHP Native Extension] Xử lý với độ trễ 0ms: %s", prompt);
    return result;
}

// Khai báo hàm PHP: evonet_infer(string prompt)
PHP_FUNCTION(evonet_infer)
{
    char *prompt;
    size_t prompt_len;

    // Phân tích tham số truyền vào từ PHP
    if (zend_parse_parameters(ZEND_NUM_ARGS(), "s", &prompt, &prompt_len) == FAILURE) {
        return;
    }

    // Gọi lõi Native
    char* output = infer_native(prompt);
    
    // Trả kết quả về cho PHP VM
    RETVAL_STRING(output);
    efree(output);
}

// Đăng ký các hàm vào Extension
const zend_function_entry evonet_functions[] = {
    PHP_FE(evonet_infer, NULL)
    PHP_FE_END
};

// Định nghĩa cấu trúc Module PHP
zend_module_entry evonet_module_entry = {
    STANDARD_MODULE_HEADER,
    "evonet",               // Tên extension
    evonet_functions,       // Danh sách các hàm
    NULL,                   // Khởi tạo Module (MINIT)
    NULL,                   // Tắt Module (MSHUTDOWN)
    NULL,                   // Khởi tạo Request (RINIT)
    NULL,                   // Tắt Request (RSHUTDOWN)
    NULL,                   // Thông tin phpinfo()
    "1.0.0",                // Version
    STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_EVONET
#ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
#endif
ZEND_GET_MODULE(evonet)
#endif
