"""runtime_tts_client 单元测试（mock HTTP，不触发真实运行时）。

覆盖（spec「数据集生成三引擎」SubTask 2.4）：
- voicedesign 成功落盘 WAV + 请求 payload 协议对齐断言
  （协议真源=CX-O-SERVER/server/qwen3_tts_provider.py：task_type/instructions/input/model）；
- cosyvoice 零样本成功：ref_audio 为 base64 data URL 列表、ref_text 列表；
- 尾部静音裁剪（阈值/最小保留语义与 provider 一致）；
- 超时 / HTTP 500 / 连接拒绝 → RuntimeTTSError（含 base_url 与排查指引）；
- 参考音频缺失 → RuntimeTTSError；
- 流式路径：跳过 44 字节 WAV 头后按配置采样率重包 WAV；
- 空音频响应 → RuntimeTTSError。

httpx.MockTransport 注入，无网络依赖；数据集目录不触碰（输出落 tmp_path）。
"""
from __future__ import annotations

import base64
import io
import json
import struct
import wave
from pathlib import Path

import httpx
import pytest

from modelstation.services.runtime_tts_client import (
    RuntimeTTSClient,
    RuntimeTTSError,
    trim_wav_tail_silence,
)

BASE_URL = "http://127.0.0.1:18099"


# ---------------------------------------------------------------------------
# 测试基建
# ---------------------------------------------------------------------------


def make_wav(samples: list[int], sample_rate: int = 24000) -> bytes:
    """构造单声道 16bit WAV 字节（测试音频素材）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def read_wav_frames(data: bytes) -> tuple[int, int, int]:
    """解析 WAV：返回 (帧数, 采样率, 声道数)。"""
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes(), wf.getframerate(), wf.getnchannels()


def make_client(handler, **overrides) -> RuntimeTTSClient:
    """构造注入 MockTransport 的客户端（各测试独立，无网络依赖）。"""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    kwargs = {
        "base_url": BASE_URL + "/",  # 带尾斜杠，验证归一化
        "model": "test-model",
        "timeout_seconds": 5.0,
        "sample_rate": 24000,
        "http_client": http,
    }
    kwargs.update(overrides)
    return RuntimeTTSClient(**kwargs)


# ---------------------------------------------------------------------------
# voicedesign 运行时
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voicedesign_success_payload_and_file(tmp_path):
    """成功合成：payload 字段与 provider 协议逐字段一致，响应字节落盘 WAV"""
    captured: dict = {}
    audio = make_wav([1000] * 500)

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, content=audio)

    client = make_client(handler)
    out = tmp_path / "vd" / "0001.wav"  # 父目录不存在，验证兜底创建
    result = await client.synthesize_voicedesign("你好世界", "温柔的女声", out)

    assert result == out
    assert out.is_file()
    assert out.read_bytes() == audio  # 无尾部静音可裁 → 原样落盘
    assert captured["url"] == f"{BASE_URL}/v1/audio/speech"
    body = captured["body"]
    assert body["model"] == "test-model"
    assert body["input"] == "你好世界"
    assert body["response_format"] == "wav"
    assert body["stream"] is False
    assert body["task_type"] == "VoiceDesign"  # vLLM 私有参数（协议真源）
    assert body["instructions"] == "温柔的女声"  # 音色描述经 instructions 传递
    assert "ref_audio" not in body  # VoiceDesign 不消费参考音频
    await client.aclose()


@pytest.mark.asyncio
async def test_voicedesign_tail_silence_trimmed(tmp_path):
    """尾部静音被裁剪：0.1s 语音 + 0.1s 静音 → 裁掉 50ms（最小保留边界），仍为合法 WAV"""
    # 2400 样本语音（0.1s）+ 2400 样本静音（0.1s）；min_retain=0.05s*24000=1200
    audio = make_wav([3000] * 2400 + [0] * 2400)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["len"] = len(audio)
        return httpx.Response(200, content=audio)

    client = make_client(handler)
    out = tmp_path / "vd_trim.wav"
    await client.synthesize_voicedesign("裁剪测试", "描述", out)

    frames, rate, _ch = read_wav_frames(out.read_bytes())
    # 保留 4800-1200=3600 帧（裁剪至最小保留边界，多声道帧对齐不适用单声道）
    assert frames == 3600
    assert rate == 24000
    await client.aclose()


def test_trim_wav_tail_silence_noop_cases():
    """全响度 / 全静音 / 非 WAV 输入 → 原样返回（不无损重写、不因裁剪失败丢数据）"""
    loud = make_wav([3000] * 1000)
    assert trim_wav_tail_silence(loud, 24000) == loud  # 无尾部静音 → 原字节
    silent = make_wav([0] * 1000)
    assert trim_wav_tail_silence(silent, 24000) == silent  # 全静音 → 原样
    assert trim_wav_tail_silence(b"not-a-wav", 24000) == b"not-a-wav"  # 解析失败 → 原样


# ---------------------------------------------------------------------------
# cosyvoice 零样本克隆运行时
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cosyvoice_zero_success_payload(tmp_path):
    """成功合成：ref_audio 为 data URL base64 列表（协议真源）、ref_text 为列表、speed/volume 恒带"""
    captured: dict = {}
    audio = make_wav([1000] * 500)
    ref_path = tmp_path / "ref.wav"
    ref_bytes = make_wav([500] * 300)
    ref_path.write_bytes(ref_bytes)

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, content=audio)

    client = make_client(handler)
    out = tmp_path / "cv" / "0001.wav"
    await client.synthesize_cosyvoice_zero("克隆这句话", ref_path, "参考转写内容", out)

    assert out.is_file()
    body = captured["body"]
    expected_b64 = base64.b64encode(ref_bytes).decode("ascii")
    assert body["ref_audio"] == [f"data:audio/wav;base64,{expected_b64}"]  # base64 内联 data URL
    assert body["ref_text"] == ["参考转写内容"]
    assert body["input"] == "克隆这句话"
    assert body["response_format"] == "wav"
    assert body["stream"] is False
    assert body["speed"] == 1.0
    assert body["volume"] == 1.0
    assert body["model"] == "test-model"
    assert "task_type" not in body  # task_type 为 voicedesign 私有参数，cosyvoice 不携带
    await client.aclose()


@pytest.mark.asyncio
async def test_cosyvoice_zero_missing_ref_audio(tmp_path):
    """参考音频不存在 → RuntimeTTSError（含路径，不发 HTTP）"""
    sent: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=b"RIFF")

    client = make_client(handler)
    with pytest.raises(RuntimeTTSError, match="参考音频不存在"):
        await client.synthesize_cosyvoice_zero("文本", tmp_path / "ghost.wav", "", tmp_path / "o.wav")
    assert not sent  # 未发出任何 HTTP 请求
    await client.aclose()


# ---------------------------------------------------------------------------
# 错误映射（超时 / HTTP 非 2xx / 连接拒绝 / 空音频）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_error_contains_base_url(tmp_path):
    """超时 → RuntimeTTSError：str 含 base_url 与「检查 vLLM 运行时是否启动」指引"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    client = make_client(handler)
    with pytest.raises(RuntimeTTSError) as ei:
        await client.synthesize_voicedesign("文本", "描述", tmp_path / "o.wav")
    assert ei.value.base_url == BASE_URL
    assert BASE_URL in str(ei.value)
    assert "检查 vLLM 运行时是否启动" in str(ei.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_http_500_error_contains_status(tmp_path):
    """HTTP 500 → RuntimeTTSError：携带状态码与原因"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal boom")

    client = make_client(handler)
    with pytest.raises(RuntimeTTSError) as ei:
        await client.synthesize_voicedesign("文本", "描述", tmp_path / "o.wav")
    assert ei.value.status_code == 500
    assert "500" in str(ei.value)
    assert BASE_URL in str(ei.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_connect_error_contains_base_url_and_hint(tmp_path):
    """连接拒绝 → RuntimeTTSError：含 base_url 与修复指引"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)
    with pytest.raises(RuntimeTTSError) as ei:
        await client.synthesize_cosyvoice_zero(
            "文本", _make_ref(tmp_path), "", tmp_path / "o.wav"
        )
    assert ei.value.base_url == BASE_URL
    assert "连接" in str(ei.value) or "不可达" in str(ei.value)
    assert "检查 vLLM 运行时是否启动" in str(ei.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_empty_audio_response_error(tmp_path):
    """200 但空音频字节 → RuntimeTTSError（空音频不入库）"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    client = make_client(handler)
    with pytest.raises(RuntimeTTSError, match="空音频"):
        await client.synthesize_voicedesign("文本", "描述", tmp_path / "o.wav")
    assert not (tmp_path / "o.wav").exists()
    await client.aclose()


# ---------------------------------------------------------------------------
# 流式路径（provider 流式处理对齐：跳 44 字节 WAV 头 → PCM 重包）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_mode_skips_wav_header_and_rewraps(tmp_path):
    """流式：响应首 44 字节为 WAV 头被跳过，剩余 PCM 按配置采样率重包 WAV 落盘"""
    pcm_samples = [2500] * 2000
    full_wav = make_wav(pcm_samples)  # 44 字节头 + 4000 字节 PCM
    assert full_wav[:4] == b"RIFF"

    async def chunked():
        # 模拟分块到达（首块不足 44 字节，验证跨块跳头）
        yield full_wav[:30]
        yield full_wav[30:100]
        yield full_wav[100:]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunked())

    client = make_client(handler)
    out = tmp_path / "stream.wav"
    await client.synthesize_voicedesign("流式文本", "描述", out, stream=True)

    data = out.read_bytes()
    frames, rate, ch = read_wav_frames(data)
    assert data[:4] == b"RIFF"  # 重包后的合法 WAV 容器
    assert frames == 2000  # PCM 样本数不变（44 字节头已剔除）
    assert rate == 24000 and ch == 1

    # 流式请求体 stream=True
    await client.aclose()


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_text_rejected_without_http(tmp_path):
    """空文本 / 空音色描述 → ValueError（不发 HTTP）"""
    sent: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=b"RIFF")

    client = make_client(handler)
    with pytest.raises(ValueError, match="text"):
        await client.synthesize_voicedesign("  ", "描述", tmp_path / "o.wav")
    with pytest.raises(ValueError, match="voice_description"):
        await client.synthesize_voicedesign("文本", "  ", tmp_path / "o.wav")
    with pytest.raises(ValueError, match="text"):
        await client.synthesize_cosyvoice_zero("", _make_ref(tmp_path), "", tmp_path / "o.wav")
    assert not sent
    await client.aclose()


def test_client_requires_base_url_and_model():
    """构造参数缺失 → ValueError（fail-fast）"""
    with pytest.raises(ValueError, match="base_url"):
        RuntimeTTSClient(base_url="", model="m")
    with pytest.raises(ValueError, match="model"):
        RuntimeTTSClient(base_url="http://x", model="")


def _make_ref(tmp_path: Path) -> Path:
    """构造测试参考音频文件。"""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(make_wav([500] * 300))
    return ref
