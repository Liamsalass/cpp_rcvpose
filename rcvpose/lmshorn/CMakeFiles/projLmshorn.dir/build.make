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
CMAKE_SOURCE_DIR = /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1

# Include any dependencies generated for this target.
include CMakeFiles/projLmshorn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/projLmshorn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/projLmshorn.dir/flags.make

CMakeFiles/projLmshorn.dir/lmshorn.c.o: CMakeFiles/projLmshorn.dir/flags.make
CMakeFiles/projLmshorn.dir/lmshorn.c.o: lmshorn.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/projLmshorn.dir/lmshorn.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/projLmshorn.dir/lmshorn.c.o   -c /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1/lmshorn.c

CMakeFiles/projLmshorn.dir/lmshorn.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/projLmshorn.dir/lmshorn.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1/lmshorn.c > CMakeFiles/projLmshorn.dir/lmshorn.c.i

CMakeFiles/projLmshorn.dir/lmshorn.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/projLmshorn.dir/lmshorn.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1/lmshorn.c -o CMakeFiles/projLmshorn.dir/lmshorn.c.s

# Object files for target projLmshorn
projLmshorn_OBJECTS = \
"CMakeFiles/projLmshorn.dir/lmshorn.c.o"

# External object files for target projLmshorn
projLmshorn_EXTERNAL_OBJECTS =

libprojLmshorn.so: CMakeFiles/projLmshorn.dir/lmshorn.c.o
libprojLmshorn.so: CMakeFiles/projLmshorn.dir/build.make
libprojLmshorn.so: CMakeFiles/projLmshorn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library libprojLmshorn.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/projLmshorn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/projLmshorn.dir/build: libprojLmshorn.so

.PHONY : CMakeFiles/projLmshorn.dir/build

CMakeFiles/projLmshorn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/projLmshorn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/projLmshorn.dir/clean

CMakeFiles/projLmshorn.dir/depend:
	cd /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1 /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1 /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1 /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1 /media/veracrypt1/Hatch/Project_3/dev/lmshorn/t1/CMakeFiles/projLmshorn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/projLmshorn.dir/depend

