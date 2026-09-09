cmake_minimum_required(VERSION 3.25)
project(seedvr2_ncnn_vulkan VERSION 0.7.0 LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
include(GNUInstallDirs)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR};$ORIGIN")
endif()
option(SEEDVR2_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
option(SEEDVR2_BUILD_WEB "Build the loopback-only local Web host" ON)

add_library(seedvr2_options INTERFACE)
target_compile_features(seedvr2_options INTERFACE cxx_std_20)
if(MSVC)
    target_compile_options(seedvr2_options INTERFACE /W4 /permissive- /utf-8)
else()
    target_compile_options(seedvr2_options INTERFACE -Wall -Wextra -Wpedantic -Wconversion)
    if(SEEDVR2_SANITIZERS)
        target_compile_options(seedvr2_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(seedvr2_options INTERFACE -fsanitize=address,undefined)
    endif()
endif()

add_library(seedvr2_planning STATIC src/planning/geometry.cpp src/planning/windows.cpp)
add_library(seedvr2::planning ALIAS seedvr2_planning)
target_include_directories(seedvr2_planning PUBLIC include)
target_link_libraries(seedvr2_planning PUBLIC seedvr2_options)

add_library(seedvr2_runtime STATIC src/runtime/run_state.cpp)
add_library(seedvr2::runtime ALIAS seedvr2_runtime)
target_include_directories(seedvr2_runtime PUBLIC include)
target_link_libraries(seedvr2_runtime PUBLIC seedvr2_options)

# Private boundary dependency. Public contracts contain only standard C++ types.
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/json.hpp" SEEDVR2_JSON_SHA)
if(NOT SEEDVR2_JSON_SHA STREQUAL "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63")
    message(FATAL_ERROR "Vendored JSON header differs from dependencies.lock.json")
endif()
add_library(seedvr2_protocol STATIC src/protocol/json.cpp)
add_library(seedvr2::protocol ALIAS seedvr2_protocol)
target_include_directories(seedvr2_protocol PUBLIC include)
target_include_directories(seedvr2_protocol SYSTEM PRIVATE third_party)
target_link_libraries(seedvr2_protocol PUBLIC seedvr2_planning)

list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.deps/install")
find_package(CLI11 2.7.2 EXACT CONFIG REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(JPEG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SEEDVR2_MEDIA REQUIRED IMPORTED_TARGET libavformat libavcodec libswscale libavutil)
find_package(SQLite3 REQUIRED)
if(TARGET SQLite3::SQLite3)
    set(SEEDVR2_SQLITE_TARGET SQLite3::SQLite3)
else()
    set(SEEDVR2_SQLITE_TARGET SQLite::SQLite3)
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS policies/seedvr2-3b.v1.json)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/policies/seedvr2-3b.v1.json" SEEDVR2_MODEL_POLICY)
configure_file(src/validation/policy.hpp.in generated/model_policy.hpp @ONLY)
add_library(seedvr2_validation STATIC src/validation/audit.cpp)
target_include_directories(seedvr2_validation PUBLIC include PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_include_directories(seedvr2_validation SYSTEM PRIVATE third_party)
target_link_libraries(seedvr2_validation PUBLIC seedvr2_options PRIVATE OpenSSL::Crypto)
add_library(seedvr2_storage STATIC src/storage/workspace.cpp)
target_include_directories(seedvr2_storage PUBLIC include)
target_include_directories(seedvr2_storage SYSTEM PRIVATE third_party)
target_link_libraries(seedvr2_storage PUBLIC seedvr2_options PRIVATE ${SEEDVR2_SQLITE_TARGET} OpenSSL::Crypto)
add_library(seedvr2_application STATIC src/application/application.cpp)
target_include_directories(seedvr2_application PUBLIC include)
target_include_directories(seedvr2_application SYSTEM PRIVATE third_party)
target_link_libraries(seedvr2_application PUBLIC seedvr2_protocol seedvr2_validation seedvr2_storage)

# Upstream source/archive are pinned and unchanged. The authenticated allocator
# overlay below is part of this application's reported runtime identity.
set(SEEDVR2_NCNN_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/.deps/ncnn-install" CACHE PATH "Prepared ncnn install")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/engine-dependencies.lock.json" SEEDVR2_ENGINE_LOCK)
string(JSON SEEDVR2_NCNN_COMMIT GET "${SEEDVR2_ENGINE_LOCK}" ncnn commit)
file(READ "${SEEDVR2_NCNN_PREFIX}/seedvr2-ncnn-commit.txt" SEEDVR2_BUILT_NCNN_COMMIT)
string(STRIP "${SEEDVR2_BUILT_NCNN_COMMIT}" SEEDVR2_BUILT_NCNN_COMMIT)
if(NOT SEEDVR2_NCNN_COMMIT STREQUAL SEEDVR2_BUILT_NCNN_COMMIT)
    message(FATAL_ERROR "Prepare the locked ncnn dependency with tools/prepare_engine.py")
endif()
find_package(ncnn CONFIG REQUIRED PATHS "${SEEDVR2_NCNN_PREFIX}/lib/cmake/ncnn" NO_DEFAULT_PATH)
if(NOT NCNN_VULKAN)
    message(FATAL_ERROR "The AWA backend requires a Vulkan-enabled ncnn build")
endif()
include(cmake/SeedVR2NcnnAllocator.cmake)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(SEEDVR2_GLSLANG_VALIDATOR glslangValidator REQUIRED)
set(SEEDVR2_AWA_SPV)
foreach(SHADER awa_gather awa_scatter awa_text_mean video_map video_norm)
    set(SPV "${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER}.spv")
    add_custom_command(OUTPUT "${SPV}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${SEEDVR2_GLSLANG_VALIDATOR}" -V --target-env vulkan1.0
            "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/ncnn/shaders/${SHADER}.comp" -o "${SPV}"
        DEPENDS "src/engine/ncnn/shaders/${SHADER}.comp" VERBATIM)
    list(APPEND SEEDVR2_AWA_SPV "${SPV}")
endforeach()
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_shaders.hpp"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_shaders.hpp" ${SEEDVR2_AWA_SPV}
    DEPENDS ${SEEDVR2_AWA_SPV} tools/embed_spirv.py VERBATIM)
configure_file(src/engine/engine_build.hpp.in generated/engine_build.hpp @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS policies/reviewed-packages.json)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/policies/reviewed-packages.json" SEEDVR2_REVIEWED_PACKAGES)
configure_file(src/engine/reviewed_packages.hpp.in generated/reviewed_packages.hpp @ONLY)
file(GLOB_RECURSE AWA_FIXTURES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/awa/*")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_fixture.cpp"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_awa_fixture.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/awa" "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_fixture.cpp"
    DEPENDS ${AWA_FIXTURES} tools/embed_awa_fixture.py VERBATIM)
add_library(seedvr2_engine SHARED src/runtime/memory_policy.cpp src/engine/ncnn/awa.cpp src/engine/ncnn/constant.cpp src/engine/ncnn/runtime.cpp
    src/engine/ncnn/package.cpp src/engine/ncnn/pipeline.cpp
    src/engine/ncnn/image.cpp src/engine/ncnn/image_io.cpp src/engine/ncnn/video_layers.cpp
    src/engine/ncnn/video.cpp src/engine/ncnn/video_io.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_fixture.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_shaders.hpp")
add_library(seedvr2::sdk ALIAS seedvr2_engine)
set_target_properties(seedvr2_engine PROPERTIES OUTPUT_NAME seedvr2 EXPORT_NAME sdk
    VERSION ${PROJECT_VERSION} SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
target_compile_features(seedvr2_engine PUBLIC cxx_std_20)
target_include_directories(seedvr2_engine PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated" src/engine/ncnn)
target_include_directories(seedvr2_engine SYSTEM PRIVATE third_party)
target_include_directories(seedvr2_engine SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.deps/src/ncnn-${SEEDVR2_NCNN_COMMIT}/src")
target_link_libraries(seedvr2_engine PRIVATE seedvr2_planning seedvr2_ncnn_runtime OpenSSL::Crypto JPEG::JPEG PkgConfig::SEEDVR2_MEDIA)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep embedded glslang/SPIRV-Tools symbols out of the dynamic symbol table.
    # Otherwise a Vulkan validation layer can bind to a different static ABI.
    target_link_options(seedvr2_engine PRIVATE "-Wl,--exclude-libs,ALL")
endif()
target_link_libraries(seedvr2_application PUBLIC seedvr2_engine)
find_package(Threads REQUIRED)
add_library(seedvr2_jobs STATIC src/jobs/service.cpp src/jobs/process.cpp)
target_include_directories(seedvr2_jobs PUBLIC include)
target_include_directories(seedvr2_jobs SYSTEM PRIVATE third_party)
target_link_libraries(seedvr2_jobs PUBLIC seedvr2_engine seedvr2_storage PRIVATE ${SEEDVR2_SQLITE_TARGET} Threads::Threads OpenSSL::Crypto)
add_executable(seedvr2 apps/cli/main.cpp)
target_include_directories(seedvr2 SYSTEM PRIVATE third_party)
target_link_libraries(seedvr2 PRIVATE seedvr2_application CLI11::CLI11)
set_target_properties(seedvr2 PROPERTIES CXX_EXTENSIONS OFF)

if(SEEDVR2_BUILD_WEB)
    find_package(Drogon 1.9.13 EXACT CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(STUDIO_DIST "${CMAKE_CURRENT_SOURCE_DIR}/apps/studio/dist")
    if(NOT EXISTS "${STUDIO_DIST}/index.html")
        message(FATAL_ERROR "Build Studio first: npm ci --prefix apps/studio && npm run build --prefix apps/studio")
    endif()
    file(GLOB_RECURSE STUDIO_ASSETS CONFIGURE_DEPENDS "${STUDIO_DIST}/*")
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/studio_assets.cpp"
        COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_studio.py"
            "${STUDIO_DIST}" "${CMAKE_CURRENT_BINARY_DIR}/generated/studio_assets.cpp"
        DEPENDS ${STUDIO_ASSETS} tools/embed_studio.py VERBATIM)
    add_executable(seedvr2-web apps/server/main.cpp apps/server/server.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/generated/studio_assets.cpp")
    target_include_directories(seedvr2-web PRIVATE apps/server)
    target_link_libraries(seedvr2-web PRIVATE seedvr2_application seedvr2_jobs Drogon::Drogon CLI11::CLI11)
    install(TARGETS seedvr2-web RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(PROGRAMS tools/seedvr2-studio DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES third_party/studio/NOTICES.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses RENAME Studio-NOTICES.txt)
endif()

add_executable(seedvr2-worker apps/worker/main.cpp)
target_include_directories(seedvr2-worker SYSTEM PRIVATE third_party)
target_link_libraries(seedvr2-worker PRIVATE seedvr2_engine CLI11::CLI11)
install(TARGETS seedvr2-worker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(BUILD_TESTING)
    add_executable(seedvr2_memory_policy_tests tests/memory_policy_tests.cpp src/runtime/memory_policy.cpp)
    target_include_directories(seedvr2_memory_policy_tests PRIVATE src/runtime)
    target_link_libraries(seedvr2_memory_policy_tests PRIVATE seedvr2_options)
    target_include_directories(seedvr2_memory_policy_tests PRIVATE include)
    add_test(NAME engine.memory-policy COMMAND seedvr2_memory_policy_tests)
    add_executable(seedvr2_memory_vulkan_tests tests/memory_vulkan_tests.cpp src/runtime/memory_policy.cpp)
    target_include_directories(seedvr2_memory_vulkan_tests PRIVATE include src/engine/ncnn)
    target_include_directories(seedvr2_memory_vulkan_tests SYSTEM PRIVATE third_party
        "${CMAKE_CURRENT_SOURCE_DIR}/.deps/src/ncnn-${SEEDVR2_NCNN_COMMIT}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/.deps/src/ncnn-${SEEDVR2_NCNN_COMMIT}/src/layer")
    target_link_libraries(seedvr2_memory_vulkan_tests PRIVATE seedvr2_ncnn_runtime OpenSSL::Crypto)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(seedvr2_memory_vulkan_tests PRIVATE "-Wl,--exclude-libs,ALL")
    endif()
    add_test(NAME engine.memory-vulkan COMMAND seedvr2_memory_vulkan_tests)
    set_tests_properties(engine.memory-vulkan PROPERTIES SKIP_RETURN_CODE 77 LABELS "vulkan" TIMEOUT 60)
    add_executable(seedvr2_awa_rope_tests tests/awa_rope_tests.cpp src/engine/ncnn/awa.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_shaders.hpp")
    target_include_directories(seedvr2_awa_rope_tests PRIVATE src/engine/ncnn "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_include_directories(seedvr2_awa_rope_tests SYSTEM PRIVATE third_party)
    target_link_libraries(seedvr2_awa_rope_tests PRIVATE seedvr2_planning seedvr2_ncnn_runtime OpenSSL::Crypto)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(seedvr2_awa_rope_tests PRIVATE "-Wl,--exclude-libs,ALL")
    endif()
    add_test(NAME engine.awa-rope-cpu COMMAND seedvr2_awa_rope_tests cpu)
    add_test(NAME engine.awa-rope-vulkan COMMAND seedvr2_awa_rope_tests vulkan)
    add_test(NAME engine.awa-sdpa-fp32-cpu COMMAND seedvr2_awa_rope_tests cpu sdpa)
    add_test(NAME engine.awa-sdpa-fp32-vulkan COMMAND seedvr2_awa_rope_tests vulkan sdpa)
    add_test(NAME engine.awa-rope-vulkan-host COMMAND seedvr2_awa_rope_tests vulkan host)
    add_test(NAME engine.awa-sdpa-fp32-vulkan-host COMMAND seedvr2_awa_rope_tests vulkan sdpa host)
    set_tests_properties(engine.awa-rope-vulkan engine.awa-sdpa-fp32-vulkan
        engine.awa-rope-vulkan-host engine.awa-sdpa-fp32-vulkan-host PROPERTIES SKIP_RETURN_CODE 77 LABELS "vulkan")
    set_tests_properties(engine.awa-rope-cpu engine.awa-rope-vulkan
        engine.awa-sdpa-fp32-cpu engine.awa-sdpa-fp32-vulkan
        engine.awa-rope-vulkan-host engine.awa-sdpa-fp32-vulkan-host PROPERTIES TIMEOUT 60)
    # Explicit developer target: real per-layer diagnostics, not a default model test.
    add_executable(seedvr2-dit-trace EXCLUDE_FROM_ALL tools/trace_dit_layers.cpp
        src/engine/ncnn/awa.cpp src/engine/ncnn/constant.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_shaders.hpp")
    target_include_directories(seedvr2-dit-trace PRIVATE src/engine/ncnn "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_include_directories(seedvr2-dit-trace SYSTEM PRIVATE third_party)
    target_link_libraries(seedvr2-dit-trace PRIVATE seedvr2_planning seedvr2_ncnn_runtime OpenSSL::Crypto)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(seedvr2-dit-trace PRIVATE "-Wl,--exclude-libs,ALL")
    endif()
    add_executable(seedvr2-vae-trace EXCLUDE_FROM_ALL tools/trace_vae_encoder.cpp
        src/engine/ncnn/video_layers.cpp "${CMAKE_CURRENT_BINARY_DIR}/generated/awa_shaders.hpp")
    target_include_directories(seedvr2-vae-trace PRIVATE src/engine/ncnn "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_include_directories(seedvr2-vae-trace SYSTEM PRIVATE third_party)
    target_link_libraries(seedvr2-vae-trace PRIVATE seedvr2_planning seedvr2_ncnn_runtime OpenSSL::Crypto)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(seedvr2-vae-trace PRIVATE "-Wl,--exclude-libs,ALL")
    endif()
    add_executable(seedvr2_image_io_tests tests/image_io_tests.cpp)
    target_include_directories(seedvr2_image_io_tests PRIVATE src/engine/ncnn)
    target_link_libraries(seedvr2_image_io_tests PRIVATE seedvr2_engine seedvr2_ncnn_runtime)
    add_test(NAME engine.image-decode COMMAND seedvr2_image_io_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/natural")
    add_executable(seedvr2_weight_io_tests tests/weight_io_tests.cpp)
    target_include_directories(seedvr2_weight_io_tests PRIVATE src/engine/ncnn)
    target_link_libraries(seedvr2_weight_io_tests PRIVATE seedvr2_ncnn_runtime seedvr2_options)
    add_test(NAME engine.mapped-reader COMMAND seedvr2_weight_io_tests)
    set_tests_properties(engine.mapped-reader PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(seedvr2_core_tests tests/core_tests.cpp)
    target_link_libraries(seedvr2_core_tests PRIVATE seedvr2_planning seedvr2_runtime seedvr2_protocol)
    target_include_directories(seedvr2_core_tests SYSTEM PRIVATE third_party)
    add_test(NAME framework.contracts COMMAND seedvr2_core_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/window-reference.json")
    add_test(NAME cli.plan COMMAND seedvr2 plan --request "${CMAKE_CURRENT_SOURCE_DIR}/examples/plan-720p.json")
    set_tests_properties(cli.plan PROPERTIES PASS_REGULAR_EXPRESSION "GEOMETRY_VALID")
    add_test(NAME worker.capability-boundary COMMAND seedvr2-worker --capabilities)
    set_tests_properties(worker.capability-boundary PROPERTIES PASS_REGULAR_EXPRESSION "seedvr2-worker-v1")
    add_test(NAME engine.awa-cpu COMMAND seedvr2 engine self-test --backend cpu)
    set_tests_properties(engine.awa-cpu PROPERTIES TIMEOUT 30)
    set_tests_properties(engine.memory-vulkan engine.awa-rope-vulkan engine.awa-sdpa-fp32-vulkan
        engine.awa-rope-vulkan-host engine.awa-sdpa-fp32-vulkan-host
        PROPERTIES FAIL_REGULAR_EXPRESSION "VUID-;Validation Error")
endif()

install(TARGETS seedvr2 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS seedvr2_engine EXPORT seedvr2Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/seedvr2/pipeline.hpp include/seedvr2/result.hpp include/seedvr2/path.hpp include/seedvr2/media.hpp include/seedvr2/memory.hpp
    include/seedvr2/image.hpp include/seedvr2/video.hpp include/seedvr2/engine.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/seedvr2)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/seedvr2Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/seedvr2Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/seedvr2)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/seedvr2ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(EXPORT seedvr2Targets NAMESPACE seedvr2:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/seedvr2)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/seedvr2Config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/seedvr2ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/seedvr2)
install(DIRECTORY examples schemas policies DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2)
install(FILES third_party/nlohmann/LICENSE.MIT DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses RENAME nlohmann-json-MIT.txt)
install(FILES tests/reference/Apache-2.0.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses RENAME SeedVR-Apache-2.0.txt)
install(FILES docs/provenance.md DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2)
install(DIRECTORY docs/ DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/docs
    FILES_MATCHING PATTERN "*.md" PATTERN "*.json" PATTERN "*.png" PATTERN "*.mp4")
install(FILES README.md README.en.md LICENSE NOTICE CONTRIBUTING.md DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2)
install(DIRECTORY artifacts/2026-09-08/delivery-v1 artifacts/2026-09-08/video-numerics-v1
    artifacts/2026-09-08/memory-v1
    DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/artifacts/2026-09-08)
install(DIRECTORY third_party/engine/ DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses/engine)
install(DIRECTORY third_party/media/ DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses/media)
install(FILES engine-dependencies.lock.json converter-dependencies.lock.json model-sources.lock.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2)

install(FILES .deps/src/cli11/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses RENAME CLI11-BSD.txt)
if(SEEDVR2_BUILD_WEB)
    install(FILES .deps/src/drogon/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses RENAME Drogon-MIT.txt)
    install(FILES .deps/src/trantor/License DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses RENAME Trantor-BSD.txt)
    install(FILES .deps/src/jsoncpp/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/seedvr2/licenses RENAME JsonCpp-MIT.txt)
endif()
