
# Add C11 flags, if C11 is available.
if (C11_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C11_FLAGS}")
endif ()

# Check if atomics are available for thread safe storage. First we try to use
# C11 atomics, next the GCC builtin atomic_fetch_add method and finally
# sync_fetch_and_add.
if (NOT C11_ATOMICS_FOUND)
  find_package(BuiltinAtomic)
  if (NOT BUILTINATOMIC_FOUND)
    find_package(BuiltinSync)
  endif ()
endif ()

configure_file(config.h.in config.h)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} # Required for generated source files.
  ${CMAKE_CURRENT_BINARY_DIR}
)


#
# Add the total module
#
add_wrapped_file(total.c total.c.w -w)
pnmpi_add_xmpi_module(timings-total "${CMAKE_CURRENT_BINARY_DIR}/total.c")

pnmpi_add_module_man(timings-total module_timings_total)

install(TARGETS timings-total DESTINATION ${PnMPI_MODULES_DIR})
