### 常用命令
- protoc mlu_op_test.proto --cpp_out=./
- protoc mlu_op_test.proto --python_out=./

### `PR` 规范

   - 标题写明任务类型，一般格式:

     ```bash
     [Prefix] Short description of the pull request (Suffix).
     ```

   - prefix：新增功能 [Feature] ，修 bug [Fix]，文档相关 [Docs] ，开发中 [WIP] （暂时不会被 review）

     跟具体算子相关修改
     ```
     [Feature](mluOpXXX): Revise code type.
     [Fix](mluOpXXX): Fix div bug.
     [Docs](mluOpXXX): Add div doc.
     ```
     跟公共模块相关修改
     ```
     [Feature](mlu-ops): Revise code type.
     [Fix](mlu-ops): Fix div bug.
     [Docs](mlu-ops): Add div doc.
     ```

   - 更多见 [PR 规范.md](https://github.com/Cambricon/mlu-ops/blob/master/docs/Pull-Request.md)
