# 私董会 AI 助手（Phase 2）

第二阶段能力：
- 会议状态控制（开始/暂停/重启/结束）
- 实时发言记录（文本模拟语音输入）
- 小虎实时提问、偏题红字提醒
- 会议结束自动纪要（核心观点/行动项/情绪观察）
- 知识库管理（导入链接或本地文档标识、删除、按会话勾选启用）
- OpenRouter 模型配置与调用（通过环境变量注入 API Key）

## 运行
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=你的key
uvicorn app:app --reload --port 8000
```

浏览器打开：`http://127.0.0.1:8000`

## API 概览
- `GET /api/session`
- `POST /api/start`
- `POST /api/pause`
- `POST /api/restart`
- `POST /api/end`
- `POST /api/speech`
- `GET /api/docs`
- `POST /api/docs`
- `DELETE /api/docs/{doc_id}`
- `POST /api/docs/select`
- `POST /api/config/openrouter`

## 说明
当前仍是演示版：
- 知识库尚未实现真正向量检索，仅做了“按会话选择文档上下文片段”
- 语音输入尚未接入实时 ASR
- 可在此基础上扩展 RAG、联网搜索、多模型路由
