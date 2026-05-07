// @ts-nocheck
import { useEffect, useMemo, useState } from 'react'
import Plot from './Plot'
import { useAppStore } from '../store'
import { getSample } from '../api'

const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
  '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]

function getVarNames(dataset: string, n: number): string[] {
  if (dataset === 'lotka_volterra') {
    const p = Math.floor(n / 2)
    return [
      ...Array.from({ length: p }, (_, i) => `Prey_${i}`),
      ...Array.from({ length: n - p }, (_, i) => `Predator_${i}`),
    ]
  }
  if (dataset === 'lorenz96') {
    return Array.from({ length: n }, (_, i) => `X_${i}`)
  }
  return Array.from({ length: n }, (_, i) => `var_${i}`)
}

export default function DataView() {
  const session = useAppStore((s) => s.session)
  const sessionInfo = useAppStore((s) => s.sessionInfo)
  const previewAdtype = useAppStore((s) => s.previewAdtype)
  const sample = useAppStore((s) => s.currentSample)
  const idx = useAppStore((s) => s.currentSampleIdx)
  const setSample = useAppStore((s) => s.setCurrentSample)
  const setIdx = useAppStore((s) => s.setCurrentSampleIdx)
  const trueCausal = useAppStore((s) => s.trueCausalMatrix)
  const runStatus = useAppStore((s) => s.runStatus)
  const results = useAppStore((s) => s.results)

  const [showCompare, setShowCompare] = useState(false)

  // 测试集大小
  const testSize = useMemo(() => {
    if (results) return results.test_size
    if (sessionInfo?.use_slice) return sessionInfo.testing_size
    return sessionInfo?.n_total_samples ?? 0
  }, [results, sessionInfo])

  // 索引切换 → 重新拉取样本
  useEffect(() => {
    if (!session) return
    let cancelled = false
    getSample(session.session_id, idx, 'auto')
      .then((s) => {
        if (!cancelled) setSample(s)
      })
      .catch(() => {/* ignore */})
    return () => {
      cancelled = true
    }
  }, [session, idx, runStatus, setSample])

  if (!session || !sample) {
    const x = Array.from({ length: 80 }, (_, i) => i)
    const demo = (() => {
      if (previewAdtype === 'step') return x.map((i) => (i < 40 ? 0.1 * Math.sin(i / 6) : 1 + 0.1 * Math.sin(i / 6)))
      if (previewAdtype === 'causal') return x.map((i) => Math.sin(i / 7) + (i > 30 && i < 50 ? (i - 30) * 0.04 : 0))
      return x.map((i) => Math.sin(i / 5) + (i === 30 ? 2 : 0) + (i === 55 ? -1.5 : 0))
    })()

    return (
      <div className="space-y-3 min-h-[560px]">
        <div className="bg-white rounded-lg shadow py-3 text-center text-slate-500">
          👈 请先在左侧准备数据集
        </div>
        <div className="glass-panel p-3">
          <div className="flex items-center justify-between mb-1">
            <h3 className="section-title mb-0 text-sm">{previewAdtype === 'spike' ? 'Spike 示例' : previewAdtype === 'step' ? 'Step 示例' : 'Causal 示例'}</h3>
            <span className="text-[11px] text-slate-500">预览</span>
          </div>
          <Plot
            data={[
              { x, y: demo, mode: 'lines', line: { color: '#10a37f' } },
            ]}
            layout={{
              height: 360,
              margin: { t: 20, b: 30, l: 40, r: 20 },
              xaxis: { title: 'Time', gridcolor: '#e5e7eb' },
              yaxis: { title: 'Value', gridcolor: '#e5e7eb' },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
            }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      </div>
    )
  }

  const varNames = getVarNames(session.dataset_name, sample.num_vars)
  const timeSteps = Array.from({ length: sample.T }, (_, i) => i)

  // ===== 正常 vs 异常对比 =====
  const compareTraces: any[] = []
  for (let v = 0; v < sample.num_vars; v++) {
    compareTraces.push({
      x: timeSteps,
      y: sample.x_n.map((row) => row[v]),
      mode: 'lines',
      name: `${varNames[v]} (正常)`,
      line: { color: COLORS[v % COLORS.length], width: 1.5, dash: 'dot' },
    })
    compareTraces.push({
      x: timeSteps,
      y: sample.x_ab.map((row) => row[v]),
      mode: 'lines',
      name: `${varNames[v]} (异常)`,
      line: { color: COLORS[v % COLORS.length], width: 2 },
    })
  }

  // ===== 异常差值 =====
  const diffTraces: any[] = []
  for (let v = 0; v < sample.num_vars; v++) {
    diffTraces.push({
      x: timeSteps,
      y: sample.x_ab.map((row, t) => row[v] - sample.x_n[t][v]),
      mode: 'lines',
      name: varNames[v],
      line: { color: COLORS[v % COLORS.length], width: 1.5 },
    })
  }

  // ===== 单独正常 / 异常 =====
  const buildTraces = (data: number[][]) =>
    data[0].map((_, v) => ({
      x: timeSteps,
      y: data.map((row) => row[v]),
      mode: 'lines',
      name: varNames[v],
      line: { color: COLORS[v % COLORS.length], width: 1.5 },
    }))

  // ===== 异常热力图 =====
  // label shape: T x num_vars，画图时转置 → variables × timesteps
  const labelHeatmap = {
    z: Array.from({ length: sample.num_vars }, (_, v) => sample.label.map((row) => row[v])),
    x: timeSteps,
    y: varNames,
    type: 'heatmap',
    colorscale: 'Blues',
    showscale: true,
    colorbar: { title: '是否异常' },
  }

  // ===== 真实因果矩阵 =====
  const causalHeatmap = trueCausal
    ? {
        z: trueCausal,
        x: varNames,
        y: varNames,
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: true,
      }
    : null

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="section-title m-0">📊 测试集样本可视化 - {session.dataset_name.toUpperCase()}</h3>
          <span className="text-xs text-slate-500">
            共 {testSize} 个测试样本{sample.from_test_set ? '（来自模型测试集）' : '（原始数据）'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-600">样本索引</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, testSize - 1)}
            value={idx}
            onChange={(e) => setIdx(parseInt(e.target.value, 10))}
            className="flex-1"
          />
          <span className="text-sm font-medium w-10 text-right">{idx}</span>
        </div>
      </div>

      <div className="glass-panel p-4 space-y-3">
        <div className="flex items-center justify-between mb-3">
          <h3 className="section-title mb-0">👁️‍🗨️ 一键可视化异常变化</h3>
          <button
            className="px-3 py-1.5 text-sm rounded bg-brand-600 text-white hover:bg-brand-700"
            onClick={() => setShowCompare((v) => !v)}
          >
            {showCompare ? '收起对比图' : '🔍 显示正常 vs 异常对比'}
          </button>
        </div>

        {showCompare && (
          <div className="space-y-4">
            <Plot
              data={compareTraces}
              layout={{
                title: '正常数据 vs 异常数据叠加对比（实线=异常，虚线=正常）',
                height: 400,
                xaxis: { title: '时间步' },
                yaxis: { title: '变量值' },
                legend: { title: { text: '变量' } },
                margin: { t: 50, b: 50, l: 60, r: 30 },
              }}
              style={{ width: '100%' }}
              useResizeHandler
              config={{ responsive: true, displayModeBar: false }}
            />
            <Plot
              data={diffTraces}
              layout={{
                title: '异常差值曲线（差值越大表示异常越明显）',
                height: 260,
                xaxis: { title: '时间步' },
                yaxis: { title: '差值' },
                margin: { t: 50, b: 50, l: 60, r: 30 },
              }}
              style={{ width: '100%' }}
              useResizeHandler
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="glass-panel p-4 space-y-3">
          <h3 className="section-title mb-0">正常时间序列</h3>
          <Plot
            data={buildTraces(sample.x_n)}
            layout={{ height: 300, xaxis: { title: '时间步' }, yaxis: { title: '值' }, margin: { t: 30, b: 40, l: 60, r: 30 } }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
        <div className="glass-panel p-4 space-y-3">
          <h3 className="section-title mb-0">异常时间序列</h3>
          <Plot
            data={buildTraces(sample.x_ab)}
            layout={{ height: 300, xaxis: { title: '时间步' }, yaxis: { title: '值' }, margin: { t: 30, b: 40, l: 60, r: 30 } }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="section-title">异常位置热力图（1 = 异常）</h3>
        <Plot
          data={[labelHeatmap as any]}
          layout={{ height: 300, xaxis: { title: '时间步' }, yaxis: { title: '变量' }, margin: { t: 30, b: 40, l: 80, r: 30 } }}
          style={{ width: '100%' }}
          useResizeHandler
          config={{ responsive: true, displayModeBar: false }}
        />
      </div>

      {causalHeatmap && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="section-title">🔗 真实因果矩阵</h3>
          {session.dataset_name === 'lotka_volterra' && (
            <p className="text-xs text-slate-500 mb-2">
              <b>生物解释</b>：前一半变量为<b>猎物 (Prey)</b>，后一半为<b>捕食者 (Predator)</b>，存在明显的捕食者-猎物因果交互。
            </p>
          )}
          {session.dataset_name === 'lorenz96' && (
            <p className="text-xs text-slate-500 mb-2">
              <b>混沌系统</b>：环形因果系统，每个变量受前两个变量影响，同时影响后两个变量。
            </p>
          )}
          <Plot
            data={[causalHeatmap as any]}
            layout={{ height: 320, xaxis: { title: '被影响变量' }, yaxis: { title: '影响变量', autorange: 'reversed' }, margin: { t: 30, b: 40, l: 80, r: 30 } }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      )}
    </div>
  )
}
