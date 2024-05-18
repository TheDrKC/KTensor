#!/usr/bin/env python

import sys
import os
import yaml
import itertools
from random import randrange

DYNAMIC = "KTENSOR_MDSPAN_NAMESPACE::dynamic_extent"

class Index:
    def __init__(self, node):
        self.symbol = node['symbol']
        try:
            self.static_extent = node['static extent']
        except KeyError:
            self.static_extent = randrange(3,7)

        try:
            self.allow_dynamic = node['allow dynamic']
        except KeyError:
            self.allow_dynamic = True

        try:
            self.forms_loop = node['forms loop']
        except KeyError:
            self.forms_loop = True

        if (self.allow_dynamic):
            self.extents_to_test = [self.static_extent, DYNAMIC]
        else:
            self.extents_to_test = [self.static_extent]

        self.declaration = f"Index<'{self.symbol}'> {self.symbol};"
        self.loop_var = self.symbol + self.symbol
        self.loop_line = f"for(int {self.loop_var}=0; {self.loop_var}<ext_{self.symbol}; ++{self.loop_var}){{"


def write(sourcefile, text, indent=2):
    """Write a line to a file with a given indentation level, adding newline
    Args:
        sourcefile: Open handle to the destination file
        text: Line to write
        indent=2: Number of spaces prepended to the line"""

    sourcefile.write(' '*indent + text + '\n')

def tensor_declaration(node,index_extents):
    """Return C++ code lines for initializing a tensor given an abstract
    description in the form of a dict (from a YAML node)
    Args:
        node: YAML node with keys 'name' and 'extents', and possibly 'scalar type'
              and/or 'tensor type', describing the tensor being declared.
        index_extents: Dict mapping index characters to their extents
    Returns:
        array_line: C++ line declaring the array being wrapped by the mdspan
                    which underlies the Tensor
        tensor_line: C++ line declaring the Tensor that wraps the array
        init_line: C++ line initializing the array values"""

    name = node['name']
    try:
        scalar_type = node['scalar type']
    except KeyError:
        scalar_type = 'int'

    try:
        tensor_type = node['tensor type']
    except KeyError:
        tensor_type = 'Tensor'

    extprod = "1"
    extents_tparams = []
    constructor_args_list = []
    has_dynamic = False
    for i in node['extents']:
        if (DYNAMIC == index_extents[i]):
            has_dynamic = True
            constructor_args_list.append(f"ext_{i}")
            extents_tparams.append(DYNAMIC)
        else:
            extents_tparams.append(f"ext_{i}")
        extprod = extprod + f"*ext_{i}"        

    if (tensor_type == 'Tensor'):
        constructor_args = f'({name}_array.data()'
        if (has_dynamic):
            array_line = f"std::vector<{scalar_type}> {name}_array({extprod});"
            constructor_args += ',' + ','.join(constructor_args_list)
        else:
            array_line = f"std::array<{scalar_type}, {extprod}> {name}_array;"

        constructor_args += ')'
        tensor_line = f"Tensor<{scalar_type}, {','.join(extents_tparams)}> {name}{constructor_args};"
        init_line = f"{name}.initialize_to_random();"
    else:
        tensor_line = f"{tensor_type}<{scalar_type}> {name};"
        init_line = ""
        array_line = ""

    return array_line, tensor_line, init_line

class Test:
    def __init__(self,testnode,test_dir):
        self.node = testnode
        self.name_underscored = testnode["name"].replace(' ','_')
        self.keys_present = testnode.keys()
        self.test_src_dir = os.path.join(test_dir,self.name_underscored)

        if ("indices" in self.keys_present):
            self.indices = [Index(node) for node in testnode['indices']]
        else:
            self.indices = []

        self.index_chars = [i.symbol for i in self.indices]
        self.extents_to_test = [i.extents_to_test for i in self.indices]

class CorrectnessTest(Test):
    def __init__(self,testnode):
        super().__init__(testnode,'correctness')

    def write_source_code_files(self):
        for prod in itertools.product(*self.extents_to_test):
            index_extents = dict(zip(self.index_chars, prod))

            try:
                os.makedirs(self.test_src_dir,exist_ok=True)
            except FileExistsError:
                # Directory already exists
                pass

            sdtag = ''.join(['d' if (p == DYNAMIC) else 's' for p in prod])

            test_src_file_name = os.path.join(self.test_src_dir, sdtag + '.cpp')
            with open(test_src_file_name,'w', encoding="utf-8") as srcfile:
                write(srcfile, "#include <array>", indent=0)
                write(srcfile, "#include <vector>", indent=0)
                write(srcfile, "#include <cstdlib>", indent=0)
                write(srcfile, "#include \"KTensor/KTensor.hpp\"\n", indent=0)
                write(srcfile, "using namespace KTensor;", indent=0)
                write(srcfile, "int main(int argc, char* argv[]){", indent=0)
                write(srcfile, "bool check = true;")

                for ki,i in enumerate(self.indices):
                    write(srcfile, i.declaration)
                    if (index_extents[i.symbol] == DYNAMIC):
                        write(srcfile, f"int const ext_{i.symbol} = std::atoi(argv[{1+ki}]);")
                    else:
                        write(srcfile, f"constexpr int ext_{i.symbol} = {index_extents[i.symbol]};")

                if ("tensors" in self.keys_present):
                    for tensor_node in self.node['tensors']:
                        array_line, tensor_line, init_line = tensor_declaration(tensor_node, index_extents)
                        write(srcfile, array_line)
                        write(srcfile, tensor_line)
                        write(srcfile, init_line)

                # Write any setup code
                if ("setup" in self.keys_present):
                    for line in self.node['setup']:
                        write(srcfile, line)

                # Write the KTensor expression whose correctness is to be checked
                expr_str = self.node['expression']
                write(srcfile, expr_str)

                # Write code used to check if the expression is implemented correctly
                # Write a for loop for each index. This can be excessive for some
                # cases, but it cuts down on what needs to be included in the
                # test definitions file and helps ensure that ranges for
                # the dimensions are correct.
                indent = 2
                for i in self.indices:
                    if (i.forms_loop):
                        write(srcfile, i.loop_line, indent=indent)
                        indent += 2

                # Write any special code that needs to go inside the check loop
                if ("inside loop" in self.keys_present):
                    for line in self.node["inside loop"]:
                        write(srcfile, line, indent=indent)

                # Write the condition to be checked
                if ("check" in self.keys_present):
                    check_str = self.node["check"].strip(';')
                else:
                    # Convert the KTensor assignment expression into a comparison
                    # between regular arithmetic expressions
                    check_str = expr_str.strip(';').replace('=', '==')
                    # Enforce floating-point division if division is involved
                    check_str = check_str.replace('/', "/(double)")
                    for i in self.indices:
                        check_str = check_str.replace(i.symbol,i.loop_var)

                write(srcfile, f"check = check && ({check_str});", indent=indent)

                # Close the for loops
                for i in self.indices:
                    if (i.forms_loop):
                        indent -= 2
                        write(srcfile, "}", indent=indent)

                # Return the error code
                write(srcfile, "if (check){")
                write(srcfile, "  return 0;")
                write(srcfile, "}else{")
                write(srcfile, "  return 1;")
                write(srcfile, "}")
                write(srcfile, "}", indent=0) # End of main function

    def write_cmake_files(self):
        with open(os.path.join(self.test_src_dir,"CMakeLists.txt"),'w') as outfile:
            for prod in itertools.product(*self.extents_to_test):
                sdtag = ''.join(['d' if (p == DYNAMIC) else 's' for p in prod])
                target_name = 'xc_' + self.name_underscored + '_' + sdtag
                ctest_name = 'test_' + self.name_underscored + '_' + sdtag

                # Add executable
                write(outfile, f"add_executable({target_name} ${{CMAKE_CURRENT_SOURCE_DIR}}/{sdtag}.cpp)")

                # Set include directories, set properties, and add tests for
                # both executables
                write(outfile,'target_include_directories(')
                write(outfile,f"    {target_name}")
                write(outfile, '    PUBLIC ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}')
                write(outfile, ')')
                # Exclude the executables from normal builds and set the C++ standard
                write(outfile,f"set_target_properties({target_name} PROPERTIES")
                write(outfile, '    EXCLUDE_FROM_ALL TRUE')
                write(outfile, '    EXCLUDE_FROM_DEFAULT_BUILD TRUE')
                write(outfile, '    CXX_STANDARD ${KTENSOR_CXX_STANDARD}')
                write(outfile, ')')
                # Add test for compiling
                write(outfile, 'add_test(')
                write(outfile,f"    NAME build_{ctest_name}")
                write(outfile,f"    COMMAND ${{CMAKE_COMMAND}} --build . --target {target_name}")
                write(outfile, '    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}')
                write(outfile, ')')
                # Add test for running
                write(outfile, 'add_test(')
                write(outfile,f"    NAME run_{ctest_name}")
                # The numbers after the executable are arguments and will set
                # any dynamic extents
                write(outfile,f"    COMMAND $<TARGET_FILE:{target_name}> 3 4 5 6 7 8 9")
                write(outfile, '    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}')
                write(outfile, ')')

class CompileTimeCheckTest(Test):
    def __init__(self,testnode):
        super().__init__(testnode,'compile_time_checks')
        
    def write_source_code_files(self):
        for prod in itertools.product(*self.extents_to_test):
            index_extents = dict(zip(self.index_chars, prod))
            
            try:
                os.makedirs(self.test_src_dir,exist_ok=True)
            except FileExistsError:
                # Directory already exists
                pass
            
            test_src_file_name = os.path.join(self.test_src_dir,''.join(['d' if (p == DYNAMIC) else 's' for p in prod]) + '.cpp')
            with open(test_src_file_name,'w', encoding="utf-8") as srcfile:
                write(srcfile, "#include \"KTensor/KTensor.hpp\"\n", indent=0)
                write(srcfile, "using namespace KTensor;\n", indent=0)
                write(srcfile, "int main(int argc, char* argv[]){", indent=0)
                # Define indices used
                for ki,i in enumerate(self.indices):
                    write(srcfile, i.declaration)
                    if (index_extents[i.symbol] == DYNAMIC):
                        write(srcfile, f"int const ext_{i.symbol} = std::atoi(argv[{1+ki}]);")
                    else:
                        write(srcfile, f"constexpr int ext_{i.symbol} = {index_extents[i.symbol]};")

                # Write any code common to the error and success cases
                # that occurs before the #ifdef
                if ('setup' in self.keys_present):
                    subkeys_present = self.node['setup'].keys()
                    for tensor_node in self.node['setup']['tensors']:
                        array_line, tensor_line, init_line = tensor_declaration(tensor_node, index_extents)
                        write(srcfile, array_line)
                        write(srcfile, tensor_line)
                    if ('lines' in subkeys_present):
                        for line in self.node['setup']['lines']:
                            write(srcfile, line)

                # Write code that will trigger the expected error
                write(srcfile, "#ifdef FORCE_ERROR", indent=0)
                subkeys_present = self.node['with bug'].keys()
                if ('tensors' in subkeys_present):
                    for tensor_node in self.node['with bug']['tensors']:
                        array_line, tensor_line, init_line = tensor_declaration(tensor_node, index_extents)
                        write(srcfile, array_line)
                        write(srcfile, tensor_line)
                if ('lines' in subkeys_present):
                    for line in self.node['with bug']['lines']:
                        write(srcfile, line)

                # Write code that should work
                subkeys_present = self.node['correct'].keys()
                write(srcfile, "#else", indent=0)
                if ('tensors' in subkeys_present):
                    for tensor_node in self.node['correct']['tensors']:
                        array_line, tensor_line, init_line = tensor_declaration(tensor_node, index_extents)
                        write(srcfile, array_line)
                        write(srcfile, tensor_line)
                if ('lines' in subkeys_present):
                    for line in self.node['correct']['lines']:
                        write(srcfile, line)
                write(srcfile, "#endif", indent=0)

                # Write any code common to both cases that occurs after the #ifdef
                if ('post' in self.keys_present):
                    for line in self.node['post']:
                        write(srcfile, line)
                # Write a return statement to prevent the compiler from optimizing
                # away all the KTensor code.
                write(srcfile, "return *(A.data_handle());")
                write(srcfile, "}", indent=0)  # End of main function

    def write_cmake_files(self):
        with open(os.path.join(self.test_src_dir,"CMakeLists.txt"),'w') as outfile:
            for prod in itertools.product(*self.extents_to_test):

                sdtag = ''.join(['d' if (p == DYNAMIC) else 's' for p in prod])
                target_name = 'xctc_' + self.name_underscored + '_' + sdtag
                ctest_name = 'test_' + self.name_underscored + '_' + sdtag

                # Add executables with and without bug
                write(outfile, f"add_executable({target_name} ${{CMAKE_CURRENT_SOURCE_DIR}}/{sdtag}.cpp)")
                write(outfile, f"add_executable({target_name}_with_bug ${{CMAKE_CURRENT_SOURCE_DIR}}/{sdtag}.cpp)")
                # Add compile definition to force a bug on the with-bug version
                write(outfile,f"target_compile_definitions({target_name}_with_bug PRIVATE FORCE_ERROR=1)")
                # Set include directories, set properties, and add tests for
                # both executables
                for bugtag in ['', '_with_bug']:
                    write(outfile,'target_include_directories(')
                    write(outfile,f"    {target_name}{bugtag}")
                    write(outfile, '    PUBLIC ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}')
                    write(outfile, ')')
                    # Exclude the executables from normal builds and set the C++ standard
                    write(outfile,f"set_target_properties({target_name}{bugtag} PROPERTIES")
                    write(outfile, '    EXCLUDE_FROM_ALL TRUE')
                    write(outfile, '    EXCLUDE_FROM_DEFAULT_BUILD TRUE')
                    write(outfile, '    CXX_STANDARD ${KTENSOR_CXX_STANDARD}')
                    write(outfile, ')')
                    # Add test
                    write(outfile, 'add_test(')
                    write(outfile,f"    NAME {ctest_name}{bugtag}")
                    write(outfile,f"    COMMAND ${{CMAKE_COMMAND}} --build . --target {target_name}{bugtag}")
                    write(outfile, '    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}')
                    write(outfile, ')')

                # Mark that the bug-version should fail
                write(outfile, 'set_property(')
                write(outfile,f"    TEST {ctest_name}_with_bug")
                write(outfile, '    PROPERTY WILL_FAIL TRUE')
                write(outfile, ')')
                
if (__name__ == "__main__"):
    for category in ['correctness', 'compile_time_checks']:

        try:
            with open(os.path.join(category,'test_definitions.yaml'), 'r', encoding="utf-8") as testfile:
                fullnode = yaml.safe_load(testfile)
        except yaml.parser.ParserError:
            print("Test definitions file contains invalid yaml")
            sys.exit(1)
        except Exception:
            print("Could not read test definitions file.")
            sys.exit(1)

        if (category == "correctness"):
            tests = [CorrectnessTest(node) for node in fullnode['tests']]

        elif (category == "compile_time_checks"):
            tests = [CompileTimeCheckTest(node) for node in fullnode['tests']]

        testdir_cmakelists_lines = []
        for test in tests:
            test.write_source_code_files()
            testdir_cmakelists_lines.append(f"add_subdirectory(\"{test.name_underscored}\")\n")
            test.write_cmake_files()

        with open(os.path.join(category,"CMakeLists.txt"),'w') as outfile:
            for line in testdir_cmakelists_lines:
                outfile.write(line)
