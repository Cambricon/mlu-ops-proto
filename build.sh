echo $PWD
protoc mlu_op_test.proto --cpp_out=.
protoc mlu_op_test.proto --python_out=./
ls -l