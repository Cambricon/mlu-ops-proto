### 常用命令
- protoc mlu_op_test.proto --cpp_out=./
- protoc mlu_op_test.proto --python_out=./

### `PR` 规范

   - 标题写明任务类型，一般格式:

     ```bash
     [Prefix] Short description of the pull request (Suffix).
     ```

   - prefix：新增功能 [Feature] ，修 bug [Fix]，文档相关 [Docs] ，开发中 [WIP] （暂时不会被 review）
     ```
     [Feature](bangc-ops): Revise code type.
     [Fix](bangc-ops): Fix div bug.
     [Docs](bangc-ops): Add div doc.
     ```

   - 更多见 [PR 规范.md]([docs/bangpy-docs/BANGPy-OPS-Operator-Development-Process.md](https://github.com/Cambricon/mlu-ops/blob/master/docs/pr)) 
