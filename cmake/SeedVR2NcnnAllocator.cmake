# Compile one authenticated compatibility copy. The upstream checkout and
# installed ncnn archive remain byte-for-byte unchanged.
set(original "${CMAKE_CURRENT_SOURCE_DIR}/.deps/src/ncnn-${SEEDVR2_NCNN_COMMIT}/src/allocator.cpp")
file(READ "${original}" contents)
string(SHA256 SEEDVR2_NCNN_ALLOCATOR_ORIGINAL_SHA "${contents}")
if(NOT SEEDVR2_NCNN_ALLOCATOR_ORIGINAL_SHA STREQUAL "601d69dab40823366fa6aa2be00c8e37bb0f1e960fe96323c36daed887c4fda2")
    message(FATAL_ERROR "Review the new ncnn allocator before applying the host-buffer compatibility fix")
endif()
set(before [=[    block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);]=])
set(after [=[    // SeedVR2 host-buffer-v1: imported memory requires matching external
    // buffer creation, and weight layout conversion may copy from this buffer.
    VkExternalMemoryBufferCreateInfo externalBufferInfo = {};
    externalBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
#if !defined(_WIN32)
    if (d->prefer_host_memory && vkdev->info.support_VK_EXT_external_memory_host())
        bufferInfo.pNext = &externalBufferInfo;
#endif
    bufferInfo.size = new_block_size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult bufferResult = vkCreateBuffer(vkdev->vkdevice(), &bufferInfo, 0, &block->buffer);
    if (bufferResult != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateBuffer for weights failed %d", bufferResult);
        delete block;
        return 0;
    }]=])
string(FIND "${contents}" "${before}" match)
if(match EQUAL -1)
    message(FATAL_ERROR "Expected ncnn weight-buffer construction is missing")
endif()
string(REPLACE "${before}" "${after}" contents "${contents}")
string(SHA256 SEEDVR2_NCNN_ALLOCATOR_PATCHED_SHA "${contents}")
set(directory "${CMAKE_CURRENT_BINARY_DIR}/generated/ncnn-host-buffer")
file(MAKE_DIRECTORY "${directory}")
file(WRITE "${directory}/allocator.cpp.in" "${contents}")
configure_file("${directory}/allocator.cpp.in" "${directory}/allocator.cpp" COPYONLY)
add_library(seedvr2_ncnn_allocator STATIC "${directory}/allocator.cpp")
target_link_libraries(seedvr2_ncnn_allocator PRIVATE ncnn)
add_library(seedvr2_ncnn_runtime INTERFACE)
# Ensure allocator.cpp.o is supplied exactly once, before the unmodified ncnn
# archive. The old allocator member is then unnecessary and is not extracted.
target_link_libraries(seedvr2_ncnn_runtime INTERFACE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,seedvr2_ncnn_allocator>" ncnn)
