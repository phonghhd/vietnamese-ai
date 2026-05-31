import os


def test_wasm_loader_syntax():
    """Kiểm tra file WASM Loader tồn tại và chứa cấu trúc chuẩn."""
    wasm_file = os.path.join(os.path.dirname(__file__), "../../sdks/wasm/evonet_wasm.js")
    assert os.path.exists(wasm_file)

    with open(wasm_file, "r") as f:
        content = f.read()

    assert "WebAssembly.Memory" in content
    assert "WebAssembly.instantiate" in content
    assert "infer(" in content


def test_node_napi_syntax():
    """Kiểm tra mã nguồn N-API C++ tồn tại và cấu hình chuẩn."""
    cpp_file = os.path.join(os.path.dirname(__file__), "../../sdks/nodejs/src/evonet_addon.cpp")
    gyp_file = os.path.join(os.path.dirname(__file__), "../../sdks/nodejs/binding.gyp")

    assert os.path.exists(cpp_file)
    assert os.path.exists(gyp_file)

    with open(cpp_file, "r") as f:
        content = f.read()

    assert "<node_api.h>" in content
    assert "NAPI_MODULE" in content
    assert "napi_create_function" in content


def test_php_zend_syntax():
    """Kiểm tra mã nguồn Zend API C tồn tại và cấu hình chuẩn."""
    c_file = os.path.join(os.path.dirname(__file__), "../../sdks/php/ext/evonet.c")

    assert os.path.exists(c_file)

    with open(c_file, "r") as f:
        content = f.read()

    assert "php.h" in content
    assert "PHP_FUNCTION" in content
    assert "zend_module_entry" in content
