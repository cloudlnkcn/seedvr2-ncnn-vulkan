#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "seedvr2::sdk" for configuration "Release"
set_property(TARGET seedvr2::sdk APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(seedvr2::sdk PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libseedvr2.so.0.7.0"
  IMPORTED_SONAME_RELEASE "libseedvr2.so.0.7"
  )

list(APPEND _cmake_import_check_targets seedvr2::sdk )
list(APPEND _cmake_import_check_files_for_seedvr2::sdk "${_IMPORT_PREFIX}/lib64/libseedvr2.so.0.7.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
