# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/renato/Documents/git-repo/LARVIO

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/renato/Documents/git-repo/LARVIO/build

# Include any dependencies generated for this target.
include CMakeFiles/flexibleInitializer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/flexibleInitializer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/flexibleInitializer.dir/flags.make

CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.o: CMakeFiles/flexibleInitializer.dir/flags.make
CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.o: ../src/FlexibleInitializer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/renato/Documents/git-repo/LARVIO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.o -c /home/renato/Documents/git-repo/LARVIO/src/FlexibleInitializer.cpp

CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/renato/Documents/git-repo/LARVIO/src/FlexibleInitializer.cpp > CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.i

CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/renato/Documents/git-repo/LARVIO/src/FlexibleInitializer.cpp -o CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.s

# Object files for target flexibleInitializer
flexibleInitializer_OBJECTS = \
"CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.o"

# External object files for target flexibleInitializer
flexibleInitializer_EXTERNAL_OBJECTS =

libflexibleInitializer.a: CMakeFiles/flexibleInitializer.dir/src/FlexibleInitializer.cpp.o
libflexibleInitializer.a: CMakeFiles/flexibleInitializer.dir/build.make
libflexibleInitializer.a: CMakeFiles/flexibleInitializer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/renato/Documents/git-repo/LARVIO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libflexibleInitializer.a"
	$(CMAKE_COMMAND) -P CMakeFiles/flexibleInitializer.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flexibleInitializer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/flexibleInitializer.dir/build: libflexibleInitializer.a

.PHONY : CMakeFiles/flexibleInitializer.dir/build

CMakeFiles/flexibleInitializer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/flexibleInitializer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/flexibleInitializer.dir/clean

CMakeFiles/flexibleInitializer.dir/depend:
	cd /home/renato/Documents/git-repo/LARVIO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/renato/Documents/git-repo/LARVIO /home/renato/Documents/git-repo/LARVIO /home/renato/Documents/git-repo/LARVIO/build /home/renato/Documents/git-repo/LARVIO/build /home/renato/Documents/git-repo/LARVIO/build/CMakeFiles/flexibleInitializer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/flexibleInitializer.dir/depend

