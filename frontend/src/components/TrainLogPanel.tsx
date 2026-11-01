/*
 * 训练详细日志面板（TrainPage / AutoModelPage 共用）：
 * 轮询 GET /api/melotts/log 展示引擎 train.log 尾部（默认 200 行），
 * 打开时每 3s 刷新并自动滚动到底部；日志缺失时给出明确提示。
 */
import { useCallback, useEffect, useRef } from "react";
import { api } from "../api/client";
import { usePolling } from "../hooks/usePolling";

export default function TrainLogPanel({ enabled }: { enabled: boolean }) {
  const { data, error } = usePolling(
    useCallback(() => api.getMelottsLog(200), []),
    { intervalMs: 3000, enabled },
  );
  const preRef = useRef<HTMLPreElement>(null);

  // 日志尾部：每次刷新后滚动到最新一行
  useEffect(() => {
    const el = preRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [data]);

  return (
    <section className="card">
      <h3 className="card-title">训练详细日志</h3>
      <p className="muted">
        来源：<span className="mono">{data?.log_path ?? "等待训练日志…"}</span>
      </p>
      {error ? <p className="muted">日志读取失败：{error}</p> : null}
      {data ? (
        data.available && data.lines.length > 0 ? (
          <pre className="log-view" ref={preRef}>
            {data.lines.join("\n")}
          </pre>
        ) : (
          <p className="muted">暂无日志（训练未启动或日志文件不存在）</p>
        )
      ) : (
        <p className="muted">读取中…</p>
      )}
    </section>
  );
}
